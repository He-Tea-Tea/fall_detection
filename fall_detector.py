# -*- coding: utf-8 -*-
"""五项分数加权与二值状态机；不再计算任何单项特征。"""

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import yaml


@dataclass
class FallEvidence:
    """main.py每帧传入的五个维度分数及恢复判断所需测量。"""

    person_id: int
    timestamp_s: float
    pose_score: float
    height_score: float
    velocity_score: float
    static_score: float
    scene_score: float
    body_angle_3d_deg: Optional[float]
    hip_height_m: Optional[float]
    data_quality: float


@dataclass
class FallDecision:
    """最终对外输出；业务状态始终只有FALL或NO_FALL。"""

    person_id: int
    is_fall: bool
    label: str
    fall_score: float
    pose_score: float
    height_score: float
    velocity_score: float
    static_score: float
    scene_score: float
    data_quality: float
    transient_evidence_active: bool


@dataclass
class _PersonMemory:
    """状态机内部计时与总分历史。"""

    history_length: int
    score_history: Deque[Tuple[float, float]] = field(init=False)
    candidate_since: Optional[float] = None
    recover_since: Optional[float] = None
    transient_evidence_until: float = 0.0
    is_fall: bool = False
    last_seen_s: float = 0.0

    def __post_init__(self) -> None:
        self.score_history = deque(maxlen=self.history_length)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


class FallDetector:
    """加权五项外部评分，并按Track ID维护FALL/NO_FALL状态。"""

    def __init__(self, config: dict):
        self.cfg = config["fall_detector"]
        self.weights = self.cfg["weights"]
        self.state_cfg = self.cfg["state_machine"]
        self.history_length = int(self.cfg["history_length"])
        self.people: Dict[int, _PersonMemory] = {}
        names = ("pose", "height", "velocity", "static", "scene")
        values = [float(self.weights[name]) for name in names]
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("fall_detector.weights必须非负且总和等于1")

    def _person(self, person_id: int) -> _PersonMemory:
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _PersonMemory(self.history_length)
        return self.people[person_id]

    @staticmethod
    def _bounded(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _update_binary_state(
        self,
        memory: _PersonMemory,
        evidence: FallEvidence,
        fall_score: float,
        transient_active: bool,
    ) -> None:
        """未跌倒时持续确认，已跌倒时持续恢复；中间计时不作为业务状态输出。"""
        now = float(evidence.timestamp_s)
        if not memory.is_fall:
            evidence_ok = evidence.pose_score >= float(self.state_cfg["min_pose_score"]) and (
                evidence.height_score >= float(self.state_cfg["min_height_score"])
                or transient_active
            )
            candidate = (
                evidence.data_quality >= float(self.state_cfg["min_data_quality"])
                and fall_score >= float(self.state_cfg["trigger_score"])
                and evidence_ok
            )
            if not candidate:
                memory.candidate_since = None
                return
            if memory.candidate_since is None:
                memory.candidate_since = now
            elif now - memory.candidate_since >= float(self.state_cfg["confirm_duration_s"]):
                memory.is_fall, memory.candidate_since, memory.recover_since = True, None, None
            return
        recovered = (
            fall_score <= float(self.state_cfg["release_score"])
            and evidence.body_angle_3d_deg is not None
            and evidence.body_angle_3d_deg <= float(self.state_cfg["recover_angle_deg"])
            and evidence.hip_height_m is not None
            and evidence.hip_height_m >= float(self.state_cfg["recover_hip_height_m"])
        )
        if not recovered:
            memory.recover_since = None
            return
        if memory.recover_since is None:
            memory.recover_since = now
        elif now - memory.recover_since >= float(self.state_cfg["recover_duration_s"]):
            memory.is_fall, memory.recover_since, memory.candidate_since = False, None, None

    def update(self, evidence: FallEvidence) -> FallDecision:
        """接收五项0～1分数，计算FallScore并更新二值状态。"""
        memory, now = self._person(evidence.person_id), float(evidence.timestamp_s)
        if memory.last_seen_s and now < memory.last_seen_s:
            raise ValueError("同一人员timestamp_s不能倒退")
        scores = {
            "pose": self._bounded(evidence.pose_score),
            "height": self._bounded(evidence.height_score),
            "velocity": self._bounded(evidence.velocity_score),
            "static": self._bounded(evidence.static_score),
            "scene": self._bounded(evidence.scene_score),
        }
        if scores["height"] >= float(self.state_cfg["transient_height_score"]) or scores[
            "velocity"
        ] >= float(self.state_cfg["transient_velocity_score"]):
            memory.transient_evidence_until = max(
                memory.transient_evidence_until,
                now + float(self.state_cfg["transient_evidence_hold_s"]),
            )
        transient_active = (
            memory.transient_evidence_until > 0.0 and now <= memory.transient_evidence_until
        )
        fall_score = sum(float(self.weights[name]) * score for name, score in scores.items())
        quality_floor = float(self.state_cfg["min_data_quality"])
        if evidence.data_quality < quality_floor:
            fall_score *= max(0.0, float(evidence.data_quality)) / max(quality_floor, 1e-8)
        fall_score = self._bounded(fall_score)
        normalized = FallEvidence(
            evidence.person_id,
            now,
            scores["pose"],
            scores["height"],
            scores["velocity"],
            scores["static"],
            scores["scene"],
            evidence.body_angle_3d_deg,
            evidence.hip_height_m,
            float(evidence.data_quality),
        )
        self._update_binary_state(memory, normalized, fall_score, transient_active)
        memory.score_history.append((now, fall_score))
        memory.last_seen_s = now
        return FallDecision(
            int(evidence.person_id),
            bool(memory.is_fall),
            "FALL" if memory.is_fall else "NO_FALL",
            fall_score,
            scores["pose"],
            scores["height"],
            scores["velocity"],
            scores["static"],
            scores["scene"],
            float(evidence.data_quality),
            transient_active,
        )

    def prune_stale(self, now_s: float) -> Sequence[int]:
        stale_after = float(self.state_cfg["stale_person_s"])
        stale_ids = [
            person_id
            for person_id, memory in self.people.items()
            if now_s - memory.last_seen_s > stale_after
        ]
        for person_id in stale_ids:
            del self.people[person_id]
        return stale_ids

    def reset_person(self, person_id: int) -> None:
        self.people.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    """直接提供五项分数，验证跌倒确认、恢复和床上躺卧不报警。"""
    detector = FallDetector(config)

    def feed(
        person_id: int,
        now: float,
        p: float,
        h: float,
        v: float,
        s: float,
        c: float,
        angle: float,
        hip: float,
    ) -> FallDecision:
        return detector.update(FallEvidence(person_id, now, p, h, v, s, c, angle, hip, 0.95))

    decision = None
    for now in np.arange(0.0, 1.0, 0.2):
        decision = feed(1, float(now), 0.0, 0.0, 0.0, 0.0, 0.5, 10.0, 0.90)
        assert not decision.is_fall
    for now in np.arange(1.0, 2.6, 0.2):
        decision = feed(1, float(now), 1.0, 1.0, 0.8 if now < 1.6 else 0.0, 1.0, 1.0, 82.0, 0.18)
    assert decision is not None and decision.is_fall
    for now in np.arange(2.6, 5.2, 0.2):
        decision = feed(1, float(now), 0.0, 0.0, 0.0, 0.0, 0.5, 10.0, 0.90)
    assert decision is not None and not decision.is_fall
    for now in np.arange(0.0, 3.0, 0.2):
        decision = feed(2, float(now), 1.0, 0.0, 0.0, 1.0, 0.0, 80.0, 0.55)
        assert not decision.is_fall
    print("fall_detector self-test: PASS")
    print("  five-score input, fall confirm, recovery, normal lying on bed=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="五项加权与二值跌倒状态机")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="统一配置文件路径",
    )
    parser.add_argument("--self-test", action="store_true", help="运行无相机合成测试")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        from main import run_live

        run_live(config, args.config, stage="fall_detector")


if __name__ == "__main__":
    main()

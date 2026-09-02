# -*- coding: utf-8 -*-
"""静止维度评分：异常姿态或低高度出现后，判断躯干3D中心是否持续低速。"""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml


@dataclass
class StaticScoreResult:
    """静止维度输出；运动速度单位为米/秒，持续时间单位为秒。"""

    score: float
    motion_velocity_mps: float
    static_duration_s: float
    candidate_active: bool
    valid_motion: bool


@dataclass
class _StaticMemory:
    history: deque
    static_since: Optional[float] = None
    last_seen_s: Optional[float] = None


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _increasing_score(value: float, start: float, full: float) -> float:
    if full <= start:
        raise ValueError("静止满分时间必须大于起始时间")
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


class StaticScorer:
    """按Track ID维护躯干中心历史和异常后静止计时。"""

    def __init__(self, config: dict):
        self.cfg = config["static_score"]
        self.history_length = int(self.cfg["history_length"])
        self.people: Dict[int, _StaticMemory] = {}

    def _person(self, person_id: int) -> _StaticMemory:
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _StaticMemory(deque(maxlen=self.history_length))
        return self.people[person_id]

    def update(
        self,
        person_id: int,
        timestamp_s: float,
        torso_center_3d: Optional[np.ndarray],
        pose_score: float,
        height_score: float,
    ) -> StaticScoreResult:
        memory, now = self._person(person_id), float(timestamp_s)
        if memory.last_seen_s is not None and now < memory.last_seen_s:
            raise ValueError("同一人员timestamp_s不能倒退")
        memory.last_seen_s = now
        candidate = float(pose_score) >= float(self.cfg["candidate_pose_score"]) or float(
            height_score
        ) >= float(self.cfg["candidate_height_score"])
        if torso_center_3d is None or not np.all(np.isfinite(torso_center_3d)):
            memory.static_since = None
            return StaticScoreResult(0.0, 0.0, 0.0, candidate, False)
        center = np.asarray(torso_center_3d, dtype=np.float64)
        previous = None
        for timestamp, old_center in reversed(memory.history):
            age = now - timestamp
            if age - float(self.cfg["motion_window_s"]) > 1e-9:
                break
            if age + 1e-9 >= float(self.cfg["min_interval_s"]):
                previous = (timestamp, old_center)
                break
        memory.history.append((now, center))
        if previous is None:
            memory.static_since = None
            return StaticScoreResult(0.0, 0.0, 0.0, candidate, False)
        timestamp, old_center = previous
        speed = float(np.linalg.norm(center - old_center) / max(now - timestamp, 1e-8))
        if not candidate or speed > float(self.cfg["motion_threshold_mps"]):
            memory.static_since = None
            return StaticScoreResult(0.0, speed, 0.0, candidate, True)
        if memory.static_since is None:
            memory.static_since = now
        duration = max(0.0, now - memory.static_since)
        score = _increasing_score(duration, float(self.cfg["start_s"]), float(self.cfg["full_s"]))
        return StaticScoreResult(score, speed, duration, True, True)

    def reset_person(self, person_id: int) -> None:
        self.people.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    scorer = StaticScorer(config)
    result = None
    for timestamp in np.arange(0.0, 3.6, 0.2):
        result = scorer.update(1, float(timestamp), np.array([0.0, 0.2, 2.0]), 1.0, 1.0)
    assert result is not None and result.valid_motion and result.score > 0.95
    moving = scorer.update(1, 3.8, np.array([0.5, 0.2, 2.0]), 1.0, 1.0)
    assert moving.score == 0.0 and moving.motion_velocity_mps > float(
        config["static_score"]["motion_threshold_mps"]
    )
    print("static self-test: PASS")
    print("  abnormal-and-still timer, movement reset=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="异常后静止维度评分")
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

        run_live(config, args.config, stage="static")


if __name__ == "__main__":
    main()

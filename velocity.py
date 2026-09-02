# -*- coding: utf-8 -*-
"""速度维度评分：根据髋部高度随时间的变化计算向下速度。"""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml


@dataclass
class VelocityScoreResult:
    """速度维度输出；负垂直速度表示向下，速度单位为米/秒。"""

    score: float
    vertical_velocity_mps: float
    downward_speed_mps: float
    valid: bool


@dataclass
class _VelocityMemory:
    history: deque
    filtered_velocity_mps: float = 0.0
    last_seen_s: Optional[float] = None


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _increasing_score(value: float, start: float, full: float) -> float:
    if full <= start:
        raise ValueError("跌倒速度阈值必须大于正常速度阈值")
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


class VelocityScorer:
    """按Track ID保存短时髋高历史并输出快速下降风险。"""

    def __init__(self, config: dict):
        self.cfg = config["velocity_score"]
        self.history_length = int(self.cfg["history_length"])
        self.people: Dict[int, _VelocityMemory] = {}

    def _person(self, person_id: int) -> _VelocityMemory:
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _VelocityMemory(deque(maxlen=self.history_length))
        return self.people[person_id]

    def update(
        self, person_id: int, timestamp_s: float, hip_height_m: Optional[float]
    ) -> VelocityScoreResult:
        memory, now = self._person(person_id), float(timestamp_s)
        if memory.last_seen_s is not None and now < memory.last_seen_s:
            raise ValueError("同一人员timestamp_s不能倒退")
        memory.last_seen_s = now
        if hip_height_m is None or not np.isfinite(hip_height_m):
            return VelocityScoreResult(0.0, memory.filtered_velocity_mps, 0.0, False)
        previous = None
        for timestamp, height in reversed(memory.history):
            age = now - timestamp
            if age - float(self.cfg["window_s"]) > 1e-9:
                break
            if age + 1e-9 >= float(self.cfg["min_interval_s"]):
                previous = (timestamp, height)
                break
        current = float(hip_height_m)
        memory.history.append((now, current))
        if previous is None:
            return VelocityScoreResult(0.0, memory.filtered_velocity_mps, 0.0, False)
        timestamp, old_height = previous
        raw_velocity = (current - old_height) / max(now - timestamp, 1e-8)
        alpha = float(self.cfg["smoothing_alpha"])
        memory.filtered_velocity_mps = (
            1.0 - alpha
        ) * memory.filtered_velocity_mps + alpha * raw_velocity
        downward = max(0.0, -memory.filtered_velocity_mps)
        score = _increasing_score(
            downward, float(self.cfg["normal_mps"]), float(self.cfg["fall_mps"])
        )
        return VelocityScoreResult(
            score, float(memory.filtered_velocity_mps), float(downward), True
        )

    def reset_person(self, person_id: int) -> None:
        self.people.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    scorer = VelocityScorer(config)
    first = scorer.update(1, 0.0, 0.90)
    falling = scorer.update(1, 0.2, 0.25)
    rising = scorer.update(1, 0.4, 0.90)
    assert (
        not first.valid
        and falling.valid
        and falling.vertical_velocity_mps < 0.0
        and falling.score > 0.5
    )
    assert rising.valid and rising.downward_speed_mps < falling.downward_speed_mps
    print("velocity self-test: PASS")
    print("  first-frame guard, downward speed, upward motion no added risk=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="下降速度维度评分")
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

        run_live(config, args.config, stage="velocity")


if __name__ == "__main__":
    main()

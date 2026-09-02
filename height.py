# -*- coding: utf-8 -*-
"""高度维度评分：髋部相对历史下降量 + 当前绝对离地高度。"""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml


@dataclass
class HeightScoreResult:
    """高度维度输出，距离单位为米。"""

    score: float
    drop_score: float
    low_height_score: float
    hip_height_m: Optional[float]
    baseline_height_m: Optional[float]
    drop_m: Optional[float]
    valid: bool


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _increasing_score(value: float, start: float, full: float) -> float:
    if full <= start:
        raise ValueError("高度下降满分阈值必须大于起始阈值")
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


def _decreasing_score(value: float, normal: float, fall: float) -> float:
    if normal <= fall:
        raise ValueError("正常髋高必须大于跌倒髋高")
    return float(np.clip((normal - value) / (normal - fall), 0.0, 1.0))


class HeightScorer:
    """按Track ID保存髋高历史并输出高度风险。"""

    def __init__(self, config: dict):
        self.cfg = config["height_score"]
        length = int(self.cfg["history_length"])
        self.histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=length))
        self.last_seen: Dict[int, float] = {}

    def update(
        self, person_id: int, timestamp_s: float, hip_height_m: Optional[float]
    ) -> HeightScoreResult:
        person_id, now = int(person_id), float(timestamp_s)
        if person_id in self.last_seen and now < self.last_seen[person_id]:
            raise ValueError("同一人员timestamp_s不能倒退")
        self.last_seen[person_id] = now
        if hip_height_m is None or not np.isfinite(hip_height_m):
            return HeightScoreResult(0.0, 0.0, 0.0, None, None, None, False)
        current = float(hip_height_m)
        candidates = [
            height
            for timestamp, height in self.histories[person_id]
            if float(self.cfg["min_baseline_age_s"])
            <= now - timestamp
            <= float(self.cfg["baseline_window_s"])
        ]
        baseline = max(candidates) if candidates else None
        drop = None if baseline is None else baseline - current
        drop_score = (
            0.0
            if drop is None
            else _increasing_score(
                drop, float(self.cfg["drop_start_m"]), float(self.cfg["drop_full_m"])
            )
        )
        low_score = _decreasing_score(
            current, float(self.cfg["hip_height_normal_m"]), float(self.cfg["hip_height_fall_m"])
        )
        self.histories[person_id].append((now, current))
        return HeightScoreResult(
            max(drop_score, low_score), drop_score, low_score, current, baseline, drop, True
        )

    def reset_person(self, person_id: int) -> None:
        self.histories.pop(int(person_id), None)
        self.last_seen.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    scorer = HeightScorer(config)
    normal = scorer.update(1, 0.0, 0.90)
    scorer.update(1, 0.2, 0.88)
    fallen = scorer.update(1, 0.4, 0.25)
    assert (
        normal.score == 0.0
        and fallen.drop_m is not None
        and fallen.drop_m > 0.5
        and fallen.score > 0.95
    )
    print("height self-test: PASS")
    print("  normal hip height, historical drop, absolute low height=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="髋部高度维度评分")
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

        run_live(config, args.config, stage="height")


if __name__ == "__main__":
    main()

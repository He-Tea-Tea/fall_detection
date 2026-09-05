# -*- coding: utf-8 -*-
"""髋部高度风险评分模块。

本文件只负责计算高度维度分H

高度维度同时判断两种情况：
    1. 相对下降：当前髋部比之前的髋部高度下降了多少。
    2. 绝对低位：当前髋部是否已经非常接近地面。

计算方法：
    baseline = 最近一段时间内的最高髋部高度
    drop = baseline - current_height
    H = max(下降量得分, 绝对低高度得分)

这样既能识别“从站立突然跌落”，也能识别“进入画面时已经躺在地面”。
所有高度和距离单位均为米，时间单位为秒。
"""

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class HeightScoreResult:
    """一次髋部高度评分结果。

    score：
        最终高度维度分H，范围为0～1。
        越接近0表示高度正常，越接近1表示高度异常。

    drop_score：
        相对历史基准高度的下降量得分。

    low_height_score：
        当前髋部绝对离地高度得分。

    hip_height_m：
        当前髋部中心到地面的距离，单位为米。

    baseline_height_m：
        历史基准髋部高度，通常取最近时间窗口内的最高值。

    drop_m：
        当前髋部相对历史基准下降的距离，单位为米。

    valid：
        True表示当前髋高有效并完成评分。
        False表示当前髋高缺失或不是有限数值。
    """

    score: float
    drop_score: float
    low_height_score: float
    hip_height_m: Optional[float]
    baseline_height_m: Optional[float]
    drop_m: Optional[float]
    valid: bool
    drop_valid: bool = False
    absolute_valid: bool = False
    jump_pending: bool = False
    reason: str = "OK"


@dataclass
class _HeightMemory:
    """一个人的已确认髋高历史和跳变候选。"""

    history_length: int
    history: Deque[Tuple[float, float]] = field(init=False)
    last_seen_s: Optional[float] = None
    last_accepted_height_m: Optional[float] = None
    last_accepted_s: Optional[float] = None
    pending_height_m: Optional[float] = None
    pending_count: int = 0
    pending_since_s: Optional[float] = None

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.history_length)


def load_config(path: str) -> dict:
    """读取统一配置文件config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _increasing_score(
    value: float,
    start: float,
    full: float,
) -> float:
    """计算数值越大、风险越高的线性分数。

    value <= start：返回0。
    value >= full：返回1。
    两个阈值之间：从0线性增加到1。
    """
    if full <= start:
        raise ValueError(
            "高度下降满分阈值必须大于起始阈值"
        )

    score = (value - start) / (full - start)
    return float(np.clip(score, 0.0, 1.0))


def _decreasing_score(
    value: float,
    normal: float,
    fall: float,
) -> float:
    """计算高度越低、风险越高的线性分数。

    value >= normal：返回0，表示髋部高度正常。
    value <= fall：返回1，表示髋部已经非常低。
    两个阈值之间：从0线性增加到1。
    """
    if normal <= fall:
        raise ValueError(
            "正常髋高必须大于跌倒髋高"
        )

    score = (normal - value) / (normal - fall)
    return float(np.clip(score, 0.0, 1.0))


class HeightScorer:
    """按照人员Track ID保存髋部高度历史并计算高度分H。

    每个人使用独立的历史记录，避免多人之间的高度数据互相影响。
    """

    def __init__(self, config: dict):
        self.cfg = config["height_score"]

        # 从config.yaml读取所有高度评分参数。
        self.history_length = int(
            self.cfg["history_length"]
        )
        self.baseline_window_s = float(
            self.cfg["baseline_window_s"]
        )
        self.min_baseline_age_s = float(
            self.cfg["min_baseline_age_s"]
        )
        self.drop_start_m = float(
            self.cfg["drop_start_m"]
        )
        self.drop_full_m = float(
            self.cfg["drop_full_m"]
        )
        self.hip_height_normal_m = float(
            self.cfg["hip_height_normal_m"]
        )
        self.hip_height_fall_m = float(
            self.cfg["hip_height_fall_m"]
        )

        # 启动时检查配置，避免运行过程中产生错误结果。
        if self.history_length <= 0:
            raise ValueError(
                "height_score.history_length必须大于0"
            )
        if self.baseline_window_s <= 0.0:
            raise ValueError(
                "height_score.baseline_window_s必须大于0"
            )
        if self.min_baseline_age_s < 0.0:
            raise ValueError(
                "height_score.min_baseline_age_s不能小于0"
            )
        if self.min_baseline_age_s > self.baseline_window_s:
            raise ValueError(
                "最小基准年龄不能大于基准时间窗口"
            )
        if self.drop_full_m <= self.drop_start_m:
            raise ValueError(
                "height_score.drop_full_m必须大于drop_start_m"
            )
        if self.hip_height_normal_m <= self.hip_height_fall_m:
            raise ValueError(
                "正常髋高必须大于跌倒髋高"
            )

        jump_cfg = self.cfg.get("jump_guard", {})
        self.jump_guard_enabled = bool(jump_cfg.get("enabled", True))
        self.max_instant_jump_m = float(jump_cfg.get("max_instant_jump_m", 0.30))
        self.jump_confirm_frames = int(jump_cfg.get("confirm_frames", 2))
        self.jump_confirm_tolerance_m = float(jump_cfg.get("confirmation_tolerance_m", 0.12))
        self.jump_max_interval_s = float(jump_cfg.get("max_interval_s", 0.60))
        if self.max_instant_jump_m <= 0.0 or self.jump_confirm_frames <= 0:
            raise ValueError("height_score.jump_guard参数必须大于0")

        # 每个人单独保存历史，避免不同Track ID互相污染。
        self.people: Dict[int, _HeightMemory] = {}

    def _person(self, person_id: int) -> _HeightMemory:
        """获取一个人的高度记忆，第一次出现时自动创建。"""
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _HeightMemory(self.history_length)
        return self.people[person_id]

    @staticmethod
    def _invalid(reason: str, jump_pending: bool = False) -> HeightScoreResult:
        """构造不会参与最终加权、也不会污染历史的无效结果。"""
        return HeightScoreResult(
            score=0.0,
            drop_score=0.0,
            low_height_score=0.0,
            hip_height_m=None,
            baseline_height_m=None,
            drop_m=None,
            valid=False,
            drop_valid=False,
            absolute_valid=False,
            jump_pending=jump_pending,
            reason=reason,
        )

    def _confirm_height(self, memory: _HeightMemory, now: float, current: float) -> bool:
        """拒绝单帧大跳变；新高度连续出现后才接受为真实运动。"""
        if (
            not self.jump_guard_enabled
            or memory.last_accepted_height_m is None
            or memory.last_accepted_s is None
            or now - memory.last_accepted_s > self.jump_max_interval_s
        ):
            memory.pending_height_m = None
            memory.pending_count = 0
            memory.pending_since_s = None
            return True
        if abs(current - memory.last_accepted_height_m) <= self.max_instant_jump_m:
            memory.pending_height_m = None
            memory.pending_count = 0
            memory.pending_since_s = None
            return True

        pending_expired = (
            memory.pending_since_s is None
            or now - memory.pending_since_s > self.jump_max_interval_s
        )
        pending_changed = (
            memory.pending_height_m is None
            or abs(current - memory.pending_height_m) > self.jump_confirm_tolerance_m
        )
        if pending_expired or pending_changed:
            memory.pending_height_m = current
            memory.pending_count = 1
            memory.pending_since_s = now
            return self.jump_confirm_frames <= 1

        memory.pending_count += 1
        memory.pending_height_m = current
        if memory.pending_count < self.jump_confirm_frames:
            return False
        memory.pending_height_m = None
        memory.pending_count = 0
        memory.pending_since_s = None
        return True

    def update(
        self,
        person_id: int,
        timestamp_s: float,
        hip_height_m: Optional[float],
        ground_valid: bool = True,
        measurement_valid: bool = True,
    ) -> HeightScoreResult:
        """更新一名人员的髋部高度并计算高度风险分H。

        参数：
            person_id：人员Track ID。
            timestamp_s：当前帧时间戳，单位为秒。
            hip_height_m：当前髋部中心到地面的距离，单位为米。

        返回：
            HeightScoreResult高度评分结果。
        """
        person_id, now = int(person_id), float(timestamp_s)
        memory = self._person(person_id)

        # 同一个人的时间必须一直向前，时间倒退会破坏历史窗口计算。
        previous_time = memory.last_seen_s
        if previous_time is not None and now < previous_time:
            raise ValueError(
                "同一人员timestamp_s不能倒退"
            )

        memory.last_seen_s = now

        if not measurement_valid:
            return self._invalid("当前3D几何或地面不可信")

        # 髋高缺失、NaN或无穷大时，不写入历史，也不编造风险分。
        if hip_height_m is None or not np.isfinite(hip_height_m):
            return self._invalid("髋高缺失")

        current_height = float(hip_height_m)
        if not self._confirm_height(memory, now, current_height):
            return self._invalid("髋高发生单帧跳变，等待连续帧确认", jump_pending=True)
        history = memory.history

        # 只选择位于规定时间窗口内、并且足够早的历史高度。
        # 排除离当前太近的数据，可以防止使用几乎相同的两帧计算下降量。
        baseline_candidates = [
            history_height
            for history_time, history_height in history
            if self.min_baseline_age_s
            <= now - history_time
            <= self.baseline_window_s
        ]

        # 使用时间窗口内的最高髋高作为跌倒前基准。
        baseline_height = (
            max(baseline_candidates)
            if baseline_candidates
            else None
        )

        # 没有足够历史时，下降量暂时无效，但绝对低高度仍然可以评分。
        drop_m = (
            baseline_height - current_height
            if baseline_height is not None
            else None
        )

        if drop_m is None:
            drop_score = 0.0
        else:
            drop_score = _increasing_score(
                drop_m,
                self.drop_start_m,
                self.drop_full_m,
            )

        # 绝对低高度强依赖地面平面；地面不可信时明确禁用这一项。
        absolute_valid = bool(ground_valid)
        low_height_score = (
            _decreasing_score(
                current_height,
                self.hip_height_normal_m,
                self.hip_height_fall_m,
            )
            if absolute_valid
            else 0.0
        )

        # 两项取最大值：
        # 有明显下降或者当前已经很低，都能为跌倒提供高度证据。
        final_score = max(
            drop_score,
            low_height_score,
        )

        # 当前样本评分完成后再写入历史，防止当前高度参与自己的基准计算。
        history.append(
            (now, current_height)
        )
        memory.last_accepted_height_m = current_height
        memory.last_accepted_s = now

        return HeightScoreResult(
            score=final_score,
            drop_score=drop_score,
            low_height_score=low_height_score,
            hip_height_m=current_height,
            baseline_height_m=baseline_height,
            drop_m=drop_m,
            valid=True,
            drop_valid=drop_m is not None,
            absolute_valid=absolute_valid,
            jump_pending=False,
            reason="OK" if absolute_valid else "地面不可信，已禁用绝对低高度",
        )

    def reset_person(self, person_id: int) -> None:
        """清除指定人员的全部高度历史。

        人员离开画面或Track ID失效后必须清除，避免以后复用相同ID时
        继承其他人员的历史髋部高度。
        """
        self.people.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    """使用合成髋高验证高度评分，不需要相机。"""
    scorer = HeightScorer(config)

    # 第1帧：人员正常站立，髋高0.90米。
    normal = scorer.update(
        person_id=1,
        timestamp_s=0.0,
        hip_height_m=0.90,
    )

    # 第2帧：轻微高度变化，不应被判断为明显下降。
    scorer.update(
        person_id=1,
        timestamp_s=0.2,
        hip_height_m=0.88,
    )

    # 第3帧：髋部下降到0.25米，应获得很高的高度风险分。
    pending = scorer.update(
        person_id=1,
        timestamp_s=0.4,
        hip_height_m=0.25,
    )
    fallen = scorer.update(
        person_id=1,
        timestamp_s=0.5,
        hip_height_m=0.26,
    )

    assert normal.valid
    assert normal.score == 0.0

    assert not pending.valid and pending.jump_pending
    assert fallen.valid and fallen.drop_valid and fallen.absolute_valid
    assert fallen.drop_m is not None
    assert fallen.drop_m > 0.5
    assert fallen.score > 0.95

    # 地面失效时仍可保留已确认的相对下降分，但绝对低高度必须关闭。
    no_absolute = scorer.update(2, 0.0, 0.20, ground_valid=False)
    assert no_absolute.valid and not no_absolute.absolute_valid
    assert no_absolute.low_height_score == 0.0

    print("height self-test: PASS")
    print(
        "  normal hip height, historical drop, "
        "jump guard, relative drop, ground-aware absolute height=PASS"
    )


def main() -> None:
    """运行合成测试或打开相机测试高度评分模块。"""
    parser = argparse.ArgumentParser(
        description="髋部高度维度评分"
    )
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config.yaml"),
        help="统一配置文件路径",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行无器相机合成测试",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.self_test:
        run_self_test(config)
    else:
        # 相机采集和可视化统一由main.py负责，本文件只计算高度分H。
        from main import run_live

        run_live(
            config,
            args.config,
            stage="height",
        )


if __name__ == "__main__":
    main()

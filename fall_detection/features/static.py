# -*- coding: utf-8 -*-
"""异常后静止风险评分模块。

本文件只负责计算静止维度分S

静止维度要回答的问题是：
    人体已经出现异常姿态或异常低高度后，是否长时间保持不动？

为什么不能看到人体静止就直接给高分：
    正常站立、坐着看电视、坐在椅子上都可能长时间静止。
    所以必须先满足以下任一异常候选条件：
        1. 姿态分P达到阈值。
        2. 高度分H达到阈值。

满足异常候选条件后，再使用躯干3D中心计算运动速度：
    speed = 当前躯干中心与历史躯干中心的距离 / 时间差

如果速度低于静止阈值：
    开始累计异常后静止时间。

如果速度超过阈值，或者姿态和高度恢复正常：
    清空静止计时。

最终静止分S：
    静止时间不足start_s：S=0。
    静止时间达到full_s：S=1。
    两个时间阈值之间：S从0线性增加到1。

注意：
    躯干中心使用Gemini深度计算，单位为米。
    速度单位为米/秒，时间单位为秒。
    摄像头移动也会导致3D中心变化，因此运行时摄像头应保持固定。
"""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class StaticScoreResult:
    """一次异常后静止评分结果。

    score：
        静止维度分S，范围为0～1。
        越接近0表示没有可靠静止证据。
        越接近1表示异常后已经长时间静止。

    motion_velocity_mps：
        躯干3D中心的空间运动速度，单位为米/秒。

    static_duration_s：
        异常候选条件下，连续保持低速的时间，单位为秒。

    candidate_active：
        当前姿态分P或高度分H是否达到异常候选阈值。

    valid_motion：
        True表示已经找到合适的历史3D中心，并成功计算运动速度。
        False表示3D中心无效或历史数据不足。
    """

    score: float
    motion_velocity_mps: float
    static_duration_s: float
    candidate_active: bool
    valid_motion: bool


@dataclass
class _StaticMemory:
    """一名人员的静止判断历史。

    history：
        保存最近若干个(timestamp, torso_center_3d)。

    static_since：
        当前连续低速状态从什么时间开始。
        None表示还没有开始累计静止时间。

    last_seen_s：
        最后一次处理该人员的时间，用于检查时间戳是否倒退。
    """

    history: Deque[Tuple[float, np.ndarray]]
    static_since: Optional[float] = None
    last_seen_s: Optional[float] = None


def load_config(path: str) -> dict:
    """读取统一配置文件config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _increasing_score(
    value: float,
    start: float,
    full: float,
) -> float:
    """把连续静止时间转换为0～1的静止风险分。

    value <= start：返回0。
    value >= full：返回1。
    两个阈值之间：从0线性增加到1。
    """
    if full <= start:
        raise ValueError(
            "静止满分时间必须大于起始时间"
        )

    score = (value - start) / (full - start)
    return float(np.clip(score, 0.0, 1.0))


class StaticScorer:
    """按照Track ID保存躯干中心历史并计算静止分S。

    每个人都有独立历史，避免不同人员的3D位置和静止时间互相影响。
    """

    def __init__(self, config: dict):
        self.cfg = config["static_score"]

        # 从config.yaml读取所有静止判断参数。
        self.history_length = int(
            self.cfg["history_length"]
        )
        self.motion_window_s = float(
            self.cfg["motion_window_s"]
        )
        self.min_interval_s = float(
            self.cfg["min_interval_s"]
        )
        self.motion_threshold_mps = float(
            self.cfg["motion_threshold_mps"]
        )
        self.start_s = float(
            self.cfg["start_s"]
        )
        self.full_s = float(
            self.cfg["full_s"]
        )
        self.candidate_pose_score = float(
            self.cfg["candidate_pose_score"]
        )
        self.candidate_height_score = float(
            self.cfg["candidate_height_score"]
        )

        # 启动时检查配置，避免运行过程中产生无法解释的结果。
        if self.history_length <= 0:
            raise ValueError(
                "static_score.history_length必须大于0"
            )
        if self.motion_window_s <= 0.0:
            raise ValueError(
                "static_score.motion_window_s必须大于0"
            )
        if self.min_interval_s <= 0.0:
            raise ValueError(
                "static_score.min_interval_s必须大于0"
            )
        if self.min_interval_s > self.motion_window_s:
            raise ValueError(
                "最小时间间隔不能大于运动计算窗口"
            )
        if self.motion_threshold_mps < 0.0:
            raise ValueError(
                "static_score.motion_threshold_mps不能小于0"
            )
        if self.full_s <= self.start_s:
            raise ValueError(
                "static_score.full_s必须大于start_s"
            )
        if not 0.0 <= self.candidate_pose_score <= 1.0:
            raise ValueError(
                "candidate_pose_score必须在0到1之间"
            )
        if not 0.0 <= self.candidate_height_score <= 1.0:
            raise ValueError(
                "candidate_height_score必须在0到1之间"
            )

        # 每个person_id对应一份独立静止历史。
        self.people: Dict[int, _StaticMemory] = {}

    def _person(
        self,
        person_id: int,
    ) -> _StaticMemory:
        """获取指定人员的历史，不存在时自动创建。"""
        person_id = int(person_id)

        if person_id not in self.people:
            self.people[person_id] = _StaticMemory(
                history=deque(
                    maxlen=self.history_length
                )
            )

        return self.people[person_id]

    def update(
        self,
        person_id: int,
        timestamp_s: float,
        torso_center_3d: Optional[np.ndarray],
        pose_score: float,
        height_score: float,
    ) -> StaticScoreResult:
        """更新一名人员的躯干位置并计算静止维度分S。

        参数：
            person_id：
                人员Track ID。

            timestamp_s：
                当前帧时间戳，单位为秒，必须单调递增。

            torso_center_3d：
                当前躯干3D中心[X, Y, Z]，单位为米。
                通常由有效肩部和髋部3D关键点的平均值得到。

            pose_score：
                姿态维度分P，范围通常为0～1。

            height_score：
                高度维度分H，范围通常为0～1。

        返回：
            StaticScoreResult静止评分结果。
        """
        person_id = int(person_id)
        now = float(timestamp_s)
        memory = self._person(person_id)

        # 同一个人的时间必须一直向前，否则速度和静止时间计算会错误。
        if (
            memory.last_seen_s is not None
            and now < memory.last_seen_s
        ):
            raise ValueError(
                "同一人员timestamp_s不能倒退"
            )

        memory.last_seen_s = now

        # 只有姿态或高度至少有一项异常，才允许开始静止判断。
        pose_abnormal = (
            float(pose_score)
            >= self.candidate_pose_score
        )
        height_abnormal = (
            float(height_score)
            >= self.candidate_height_score
        )
        candidate_active = (
            pose_abnormal
            or height_abnormal
        )

        # 躯干3D中心缺失时无法计算运动速度。
        # 此时清除连续静止计时，防止数据缺失被误认为人体没有运动。
        if torso_center_3d is None:
            memory.static_since = None
            return StaticScoreResult(
                score=0.0,
                motion_velocity_mps=0.0,
                static_duration_s=0.0,
                candidate_active=candidate_active,
                valid_motion=False,
            )

        center = np.asarray(
            torso_center_3d,
            dtype=np.float64,
        )

        if center.shape != (3,):
            raise ValueError(
                "torso_center_3d必须是包含[X, Y, Z]的三维坐标"
            )

        if not np.all(np.isfinite(center)):
            memory.static_since = None
            return StaticScoreResult(
                score=0.0,
                motion_velocity_mps=0.0,
                static_duration_s=0.0,
                candidate_active=candidate_active,
                valid_motion=False,
            )

        # 在最近motion_window_s时间内寻找一个历史3D中心。
        # 历史点必须与当前帧至少间隔min_interval_s，避免时间差太小
        # 导致轻微Depth抖动被除以极小时间后变成很大的速度。
        previous = None

        for history_time, old_center in reversed(
            memory.history
        ):
            age = now - history_time

            # reversed()从最新历史开始查找。
            # 当前历史点已经超过运动窗口时，更早的数据也会超出窗口。
            if age - self.motion_window_s > 1e-9:
                break

            if age + 1e-9 >= self.min_interval_s:
                previous = (
                    history_time,
                    old_center,
                )
                break

        # 当前有效位置先写入历史，供后面的帧计算运动速度。
        memory.history.append(
            (now, center.copy())
        )

        # 第一帧或历史时间间隔不足时，暂时不能计算运动速度。
        if previous is None:
            memory.static_since = None
            return StaticScoreResult(
                score=0.0,
                motion_velocity_mps=0.0,
                static_duration_s=0.0,
                candidate_active=candidate_active,
                valid_motion=False,
            )

        history_time, old_center = previous
        time_interval = max(
            now - history_time,
            1e-8,
        )

        # 计算躯干3D中心的空间位移。
        displacement_m = float(
            np.linalg.norm(
                center - old_center
            )
        )

        # 运动速度 = 三维空间位移 / 时间差。
        motion_velocity_mps = (
            displacement_m / time_interval
        )

        # 以下任一情况出现时，不能继续累计静止时间：
        # 1. 姿态分P和高度分H都没有达到异常候选阈值。
        # 2. 躯干运动速度超过静止阈值。
        moving = (
            motion_velocity_mps
            > self.motion_threshold_mps
        )

        if not candidate_active or moving:
            memory.static_since = None

            return StaticScoreResult(
                score=0.0,
                motion_velocity_mps=motion_velocity_mps,
                static_duration_s=0.0,
                candidate_active=candidate_active,
                valid_motion=True,
            )

        # 到这里说明：
        # 1. 姿态或高度已经异常。
        # 2. 躯干运动速度低于静止阈值。
        # 第一次满足条件时记录静止开始时间。
        if memory.static_since is None:
            memory.static_since = now

        static_duration_s = max(
            0.0,
            now - memory.static_since,
        )

        # 静止时间达到start_s后开始得分，达到full_s后记满分。
        score = _increasing_score(
            static_duration_s,
            self.start_s,
            self.full_s,
        )

        return StaticScoreResult(
            score=score,
            motion_velocity_mps=motion_velocity_mps,
            static_duration_s=static_duration_s,
            candidate_active=True,
            valid_motion=True,
        )

    def reset_person(
        self,
        person_id: int,
    ) -> None:
        """清除指定人员的全部静止历史。

        人员离开画面或Track ID失效后必须清除，避免以后复用相同ID时
        继承其他人员的躯干位置和静止时间。
        """
        self.people.pop(
            int(person_id),
            None,
        )


def run_self_test(config: dict) -> None:
    """使用合成3D中心验证静止累计和运动重置，不需要相机。"""
    scorer = StaticScorer(config)
    result = None

    # 模拟一个姿态和高度都异常，但躯干中心持续不动的人。
    # 躯干位置始终为[0.0, 0.2, 2.0]米。
    for timestamp in np.arange(
        0.0,
        3.6,
        0.2,
    ):
        result = scorer.update(
            person_id=1,
            timestamp_s=float(timestamp),
            torso_center_3d=np.array(
                [0.0, 0.2, 2.0],
                dtype=np.float64,
            ),
            pose_score=1.0,
            height_score=1.0,
        )

    # 持续静止足够长后，静止分应该接近1。
    assert result is not None
    assert result.valid_motion
    assert result.score > 0.95

    # 模拟躯干中心突然从X=0.0移动到X=0.5米。
    # 运动速度应该超过阈值，并立即清空静止分。
    moving = scorer.update(
        person_id=1,
        timestamp_s=3.8,
        torso_center_3d=np.array(
            [0.5, 0.2, 2.0],
            dtype=np.float64,
        ),
        pose_score=1.0,
        height_score=1.0,
    )

    assert moving.score == 0.0
    assert (
        moving.motion_velocity_mps
        > float(
            config["static_score"][
                "motion_threshold_mps"
            ]
        )
    )

    print("static self-test: PASS")
    print(
        "  abnormal-and-still timer, "
        "movement reset=PASS"
    )


def main() -> None:
    """运行无相机测试或打开静止维度测试窗口。"""
    parser = argparse.ArgumentParser(
        description="异常后静止维度评分"
    )
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config.yaml"),
        help="统一配置文件路径",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行无相机合成测试",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.self_test:
        run_self_test(config)
    else:
        # 相机采集、3D反投影和可视化由main.py统一负责。
        # 本文件只接收躯干3D中心、姿态分P和高度分H。
        from ..app.main import run_live

        run_live(
            config,
            args.config,
            stage="static",
        )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""下降速度维度评分模块。

本文件只负责计算五维跌倒评分中的速度分V，不判断最终是否跌倒。

数据来源：
1. main.py从YOLO Pose获得左右髋关键点；
2. 从对齐到RGB画面的Depth中读取髋部深度；
3. 将左右髋关键点反投影成3D坐标；
4. 计算髋中心到地面的距离，得到hip_height_m；
5. 本模块比较同一Track ID前后两次髋高，计算垂直速度。

计算公式：
vertical_velocity = (当前髋高 - 过去髋高) / 时间差

速度含义：
- vertical_velocity < 0：髋部向下运动；
- vertical_velocity > 0：髋部向上运动；
- downward_speed = max(0, -vertical_velocity)。

最后把向下速度转换成0～1的风险分V：
- 向下速度不超过normal_mps时，V=0；
- 向下速度达到fall_mps时，V=1；
- 两个阈值之间采用线性变化。

第一帧、Depth缺失或找不到合适历史点时，valid=False。此时score=0只是
占位值，fall_detector.py应根据valid字段忽略该维度，不能把它当成正常0分。
"""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import yaml


@dataclass
class VelocityScoreResult:
    """一次速度维度计算结果。

    vertical_velocity_mps：
        平滑后的垂直速度，负数表示向下，正数表示向上。

    downward_speed_mps：
        只保留向下速度；向上运动时该值为0。

    valid：
        True表示本帧成功找到历史高度并计算出速度；
        False表示数据不足，score不能参与最终加权。
    """

    score: float
    vertical_velocity_mps: float
    downward_speed_mps: float
    valid: bool


@dataclass
class _VelocityMemory:
    """一个Track ID对应的内部速度历史。"""

    # 保存若干个(时间, 髋高)数据，髋高单位为米。
    history: Deque[Tuple[float, float]]

    # 指数平滑后的垂直速度，单位为米/秒。
    filtered_velocity_mps: float = 0.0

    # 用于检查同一个人的时间是否发生倒退。
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
    """将数值转换成0～1递增风险分。

    value <= start时返回0；
    value >= full时返回1；
    中间范围使用线性插值。
    """
    if full <= start:
        raise ValueError("跌倒速度阈值必须大于正常速度阈值")

    score = (value - start) / (full - start)
    return float(np.clip(score, 0.0, 1.0))


class VelocityScorer:
    """按Track ID保存短时髋高历史，并计算下降速度风险V。"""

    def __init__(self, config: dict):
        self.cfg = config["velocity_score"]
        self.history_length = int(self.cfg["history_length"])
        self.window_s = float(self.cfg["window_s"])
        self.min_interval_s = float(self.cfg["min_interval_s"])
        self.normal_mps = float(self.cfg["normal_mps"])
        self.fall_mps = float(self.cfg["fall_mps"])
        self.smoothing_alpha = float(self.cfg["smoothing_alpha"])
        self.people: Dict[int, _VelocityMemory] = {}

        # 启动时检查配置，避免运行过程中一直得不到有效速度。
        if self.history_length <= 0:
            raise ValueError("velocity_score.history_length必须大于0")
        if self.window_s <= 0.0:
            raise ValueError("velocity_score.window_s必须大于0")
        if self.min_interval_s <= 0.0:
            raise ValueError("velocity_score.min_interval_s必须大于0")
        if self.min_interval_s > self.window_s:
            raise ValueError(
                "velocity_score.min_interval_s不能大于window_s"
            )
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError(
                "velocity_score.smoothing_alpha必须大于0且不超过1"
            )

        # 调用一次评分函数，检查速度阈值是否正确。
        _increasing_score(
            0.0,
            self.normal_mps,
            self.fall_mps,
        )

    def _person(self, person_id: int) -> _VelocityMemory:
        """获取指定人员的历史；第一次出现时自动创建。"""
        person_id = int(person_id)

        if person_id not in self.people:
            self.people[person_id] = _VelocityMemory(
                history=deque(maxlen=self.history_length)
            )

        return self.people[person_id]

    def _find_previous_height(
        self,
        memory: _VelocityMemory,
        now: float,
    ) -> Optional[Tuple[float, float]]:
        """在指定时间窗口内寻找一个合适的过去髋高。

        历史从新到旧搜索，选择距离当前时间至少min_interval_s，
        但不超过window_s的最近数据。

        时间间隔太短时，几毫米Depth抖动也可能被除成很大的速度；
        时间间隔太长时，速度又不能代表当前快速下降过程。
        """
        for timestamp, height in reversed(memory.history):
            age = now - timestamp

            # 历史按时间顺序保存；当前点已超过窗口后，更旧的点也不再可用。
            if age > self.window_s + 1e-9:
                break

            if age >= self.min_interval_s - 1e-9:
                return timestamp, height

        return None

    def update(
        self,
        person_id: int,
        timestamp_s: float,
        hip_height_m: Optional[float],
    ) -> VelocityScoreResult:
        """输入当前髋高，返回该人员本帧的速度风险分V。

        参数：
        person_id：
            YOLO跟踪器提供的稳定Track ID，不同人员不能共用历史。

        timestamp_s：
            当前单调时间，单位秒，通常由main.py的time.monotonic()提供。

        hip_height_m：
            髋中心到地面平面的距离，单位米；
            Depth缺失或髋部3D无效时传入None。
        """
        person_id = int(person_id)
        now = float(timestamp_s)
        memory = self._person(person_id)

        # 时间倒退会导致负时间差和错误速度，因此直接报错。
        if (
            memory.last_seen_s is not None
            and now < memory.last_seen_s
        ):
            raise ValueError("同一人员timestamp_s不能倒退")

        memory.last_seen_s = now

        # 髋高无效时不写入历史，避免错误Depth污染后续速度。
        if hip_height_m is None or not np.isfinite(hip_height_m):
            return VelocityScoreResult(
                score=0.0,
                vertical_velocity_mps=0.0,
                downward_speed_mps=0.0,
                valid=False,
            )

        current_height = float(hip_height_m)
        previous = self._find_previous_height(memory, now)

        # 当前有效髋高要保存，供后续帧计算速度。
        memory.history.append((now, current_height))

        # 第一帧或长时间断流后没有合适历史点，无法计算速度。
        if previous is None:
            # 清除旧平滑速度，防止断流前的下降速度影响新数据。
            memory.filtered_velocity_mps = 0.0
            return VelocityScoreResult(
                score=0.0,
                vertical_velocity_mps=0.0,
                downward_speed_mps=0.0,
                valid=False,
            )

        previous_time, previous_height = previous
        time_interval = max(now - previous_time, 1e-8)

        # 当前高度小于过去高度时，raw_velocity为负数，表示向下运动。
        raw_velocity = (
            current_height - previous_height
        ) / time_interval

        # 指数平滑：
        # 新速度 = (1-alpha)×旧速度 + alpha×当前原始速度。
        # alpha越大响应越快，但Depth噪声也越明显。
        alpha = self.smoothing_alpha
        memory.filtered_velocity_mps = (
            (1.0 - alpha) * memory.filtered_velocity_mps
            + alpha * raw_velocity
        )

        # 最终评分只关心向下运动；向上或静止不增加下降风险。
        downward_speed = max(
            0.0,
            -memory.filtered_velocity_mps,
        )
        score = _increasing_score(
            downward_speed,
            self.normal_mps,
            self.fall_mps,
        )

        return VelocityScoreResult(
            score=score,
            vertical_velocity_mps=float(
                memory.filtered_velocity_mps
            ),
            downward_speed_mps=float(downward_speed),
            valid=True,
        )

    def reset_person(self, person_id: int) -> None:
        """人员离场后清除历史，防止Track ID复用旧速度。"""
        self.people.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    """使用合成髋高验证首次测量、下降、上升和断流保护。"""
    scorer = VelocityScorer(config)

    # 第一帧只有当前高度，没有过去高度，所以速度无效。
    first = scorer.update(
        person_id=1,
        timestamp_s=0.0,
        hip_height_m=0.90,
    )

    # 0.2秒内从0.90米降到0.25米，应产生明显向下速度。
    falling = scorer.update(
        person_id=1,
        timestamp_s=0.2,
        hip_height_m=0.25,
    )

    # 又从0.25米升到0.90米，向下风险应明显降低。
    rising = scorer.update(
        person_id=1,
        timestamp_s=0.4,
        hip_height_m=0.90,
    )

    assert not first.valid
    assert falling.valid
    assert falling.vertical_velocity_mps < 0.0
    assert falling.downward_speed_mps > 0.0
    assert falling.score > 0.5
    assert rising.valid
    assert (
        rising.downward_speed_mps
        < falling.downward_speed_mps
    )

    # 模拟超过速度窗口的长时间断流。
    # 重新出现后的第一帧不能继承断流前的平滑速度。
    after_gap = scorer.update(
        person_id=1,
        timestamp_s=1.0,
        hip_height_m=0.90,
    )
    assert not after_gap.valid
    assert after_gap.vertical_velocity_mps == 0.0

    # 新序列第二个有效高度与上一个相同，应得到0下降速度。
    stable = scorer.update(
        person_id=1,
        timestamp_s=1.2,
        hip_height_m=0.90,
    )
    assert stable.valid
    assert stable.downward_speed_mps == 0.0
    assert stable.score == 0.0

    # None表示Depth或髋部3D缺失，不能参与速度计算。
    missing = scorer.update(
        person_id=2,
        timestamp_s=0.0,
        hip_height_m=None,
    )
    assert not missing.valid
    assert missing.score == 0.0

    print("velocity self-test: PASS")
    print(
        "  first-frame guard, downward speed, rising motion, "
        "missing-depth guard, stale-filter reset=PASS"
    )


def main() -> None:
    """独立运行入口：--self-test测试，默认调用main.py打开相机。"""
    parser = argparse.ArgumentParser(
        description="下降速度维度评分"
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent / "config.yaml"
        ),
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
        # 相机只由main.py统一打开，本文件复用完整在线流程。
        from main import run_live

        run_live(
            config,
            args.config,
            stage="velocity",
        )


if __name__ == "__main__":
    main()
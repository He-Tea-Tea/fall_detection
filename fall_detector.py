# -*- coding: utf-8 -*-
"""跌倒评分与二值状态机。

职责边界
--------
本文件的算法不运行相机、YOLO、3D反投影、人体角度或场景关系识别。它只
消费已经计算好的结构化测量，完成五项评分：

1. pose：人体角度 + 人框宽高比；
2. height：相对历史高度下降 + 当前髋部低位；
3. velocity：髋部向下速度；
4. static：异常姿态后持续低速；
5. scene：地面高风险，床/沙发/椅子低风险。

状态机对外只有两种结果：FALL或NO_FALL。候选计时和恢复计时都是内部变量，
不会再向上层暴露SUSPECT、FALLING、RECOVERED等业务状态。直接执行本文件
时，会调用main.py提供的公共相机测试流程显示评分和二值结果。
"""

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

@dataclass
class FallObservation:
    """main.py每帧传入的一人测量，所有3D单位均为米。"""

    person_id: int
    timestamp_s: float
    body_angle_deg: Optional[float]
    bbox_width_height_ratio: float
    hip_height_m: Optional[float]
    torso_center_3d: Optional[np.ndarray]
    scene_name: str
    data_quality: float

@dataclass
class FallDecision:
    """跌倒模块对外输出；is_fall是唯一状态结果。"""

    person_id: int
    is_fall: bool
    label: str
    fall_score: float
    pose_score: float
    height_score: float
    velocity_score: float
    static_score: float
    scene_score: float
    vertical_velocity_mps: float
    motion_velocity_mps: float
    static_duration_s: float
    data_quality: float
    transient_evidence_active: bool

@dataclass
class _PersonMemory:
    """二值状态机内部历史，不作为公共业务状态。"""

    history_length: int
    height_history: Deque[Tuple[float, float]] = field(init=False)
    center_history: Deque[Tuple[float, np.ndarray]] = field(init=False)
    score_history: Deque[Tuple[float, float]] = field(init=False)
    vertical_velocity_mps: float = 0.0
    motion_velocity_mps: float = 0.0
    static_since: Optional[float] = None
    candidate_since: Optional[float] = None
    recover_since: Optional[float] = None
    transient_evidence_until: float = 0.0
    is_fall: bool = False
    last_seen_s: float = 0.0

    def __post_init__(self) -> None:
        """按统一历史长度创建髋高、躯干中心和总分队列。"""
        self.height_history = deque(maxlen=self.history_length)
        self.center_history = deque(maxlen=self.history_length)
        self.score_history = deque(maxlen=self.history_length)

def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def _increasing_score(value: float, start: float, full: float) -> float:
    """数值越大风险越高的线性0~1归一化。"""
    if full <= start:
        raise ValueError("评分满分阈值必须大于起始阈值")
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))

def _decreasing_score(value: float, normal: float, fall: float) -> float:
    """数值越小风险越高的线性0~1归一化。"""
    if normal <= fall:
        raise ValueError("正常高度阈值必须大于跌倒高度阈值")
    return float(np.clip((normal - value) / (normal - fall), 0.0, 1.0))

class FallDetector:
    """按人员Track ID维护评分历史，并输出二值跌倒结果。"""

    def __init__(self, config: dict):
        """读取五项评分与二值状态机参数，每个Track ID随后独立维护历史。"""
        self.cfg = config["fall_detector"]
        self.weights = self.cfg["weights"]
        self.pose_cfg = self.cfg["pose_score"]
        self.height_cfg = self.cfg["height_score"]
        self.velocity_cfg = self.cfg["velocity_score"]
        self.static_cfg = self.cfg["static_score"]
        self.scene_scores = self.cfg["scene_scores"]
        self.state_cfg = self.cfg["state_machine"]
        self.history_length = int(self.cfg["history_length"])
        self.people: Dict[int, _PersonMemory] = {}
        self._validate_weights()

    def _validate_weights(self) -> None:
        """五项权重必须非负且总和为1，避免FallScore含义漂移。"""
        names = ("pose", "height", "velocity", "static", "scene")
        values = [float(self.weights[name]) for name in names]
        if any(value < 0.0 for value in values):
            raise ValueError("fall_detector.weights不能为负数")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("fall_detector.weights总和必须等于1")

    def _person(self, person_id: int) -> _PersonMemory:
        """首次看到Track ID时创建独立历史。"""
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _PersonMemory(self.history_length)
        return self.people[person_id]

    def _pose_score(self, angle_deg: Optional[float], bbox_ratio: float) -> float:
        """融合人体方向角和人框横宽竖高程度。"""
        if angle_deg is None:
            angle_score = 0.0
        else:
            angle_score = _increasing_score(
                float(angle_deg),
                float(self.pose_cfg["angle_normal_deg"]),
                float(self.pose_cfg["angle_fall_deg"]),
            )
        shape_score = _increasing_score(
            float(bbox_ratio),
            float(self.pose_cfg["bbox_ratio_normal"]),
            float(self.pose_cfg["bbox_ratio_fall"]),
        )
        angle_weight = float(self.pose_cfg["angle_weight"])
        return float(np.clip(angle_weight * angle_score + (1.0 - angle_weight) * shape_score, 0.0, 1.0))

    def _height_score(self, memory: _PersonMemory, now: float, hip_height_m: Optional[float]) -> float:
        """取历史站立高度下降量和当前绝对低位中的较大风险。"""
        if hip_height_m is None:
            return 0.0
        current = float(hip_height_m)
        baseline_candidates = [
            height
            for timestamp, height in memory.height_history
            if float(self.height_cfg["min_baseline_age_s"]) <= now - timestamp
            <= float(self.height_cfg["baseline_window_s"])
        ]
        drop_score = 0.0
        if baseline_candidates:
            baseline = max(baseline_candidates)
            drop_score = _increasing_score(
                baseline - current,
                float(self.height_cfg["drop_start_m"]),
                float(self.height_cfg["drop_full_m"]),
            )
        low_height_score = _decreasing_score(
            current,
            float(self.height_cfg["hip_height_normal_m"]),
            float(self.height_cfg["hip_height_fall_m"]),
        )
        return max(drop_score, low_height_score)

    def _vertical_velocity(self, memory: _PersonMemory, now: float, hip_height_m: Optional[float]) -> float:
        """根据一定时间间隔前的髋高计算垂直速度，向下为负。"""
        if hip_height_m is None:
            return memory.vertical_velocity_mps
        previous = None
        for timestamp, height in reversed(memory.height_history):
            age = now - timestamp
            if age >= float(self.velocity_cfg["min_interval_s"]):
                previous = (timestamp, height)
                break
            if age > float(self.velocity_cfg["window_s"]):
                break
        if previous is None:
            return memory.vertical_velocity_mps
        timestamp, old_height = previous
        dt = now - timestamp
        if dt <= 1e-8 or dt > float(self.velocity_cfg["window_s"]):
            return memory.vertical_velocity_mps
        raw = (float(hip_height_m) - old_height) / dt
        alpha = float(self.velocity_cfg["smoothing_alpha"])
        memory.vertical_velocity_mps = float((1.0 - alpha) * memory.vertical_velocity_mps + alpha * raw)
        return memory.vertical_velocity_mps

    def _velocity_score(self, vertical_velocity_mps: float) -> float:
        """只给向下运动计分；站起时的正速度不增加跌倒风险。"""
        downward_speed = max(0.0, -float(vertical_velocity_mps))
        return _increasing_score(downward_speed, float(self.velocity_cfg["normal_mps"]), float(self.velocity_cfg["fall_mps"]),)

    def _motion_velocity(self, memory: _PersonMemory, now: float, center_3d: Optional[np.ndarray]) -> float:
        """计算躯干中心三维速度，用于判断倒地后是否持续静止。"""
        if center_3d is None or not np.all(np.isfinite(center_3d)):
            return memory.motion_velocity_mps
        center = np.asarray(center_3d, dtype=np.float64)
        previous = None
        for timestamp, old_center in reversed(memory.center_history):
            age = now - timestamp
            if age >= float(self.static_cfg["min_interval_s"]):
                previous = (timestamp, old_center)
                break
            if age > float(self.static_cfg["motion_window_s"]):
                break
        if previous is not None:
            timestamp, old_center = previous
            dt = now - timestamp
            if 1e-8 < dt <= float(self.static_cfg["motion_window_s"]):
                memory.motion_velocity_mps = float(np.linalg.norm(center - old_center) / dt)
        return memory.motion_velocity_mps

    def _static_score(
        self,
        memory: _PersonMemory,
        now: float,
        pose_score: float,
        height_score: float,
        motion_velocity_mps: float,
    ) -> Tuple[float, float]:
        """仅在姿态或高度已异常时累计低速持续时间。"""
        candidate = (
            pose_score >= float(self.static_cfg["candidate_pose_score"])
            or height_score >= float(self.static_cfg["candidate_height_score"])
        )
        if not candidate or motion_velocity_mps > float(self.static_cfg["motion_threshold_mps"]):
            memory.static_since = None
            return 0.0, 0.0
        if memory.static_since is None:
            memory.static_since = now
        duration = max(0.0, now - memory.static_since)
        score = _increasing_score(duration, float(self.static_cfg["start_s"]), float(self.static_cfg["full_s"]),)
        return score, duration

    def _scene_score(self, scene_name: str) -> float:
        """把sense.py输出的主场景映射为风险分。"""
        return float(self.scene_scores.get(scene_name, self.scene_scores["unknown"]))

    def _update_binary_state(
        self,
        memory: _PersonMemory,
        observation: FallObservation,
        fall_score: float,
        pose_score: float,
        height_score: float,
        transient_active: bool,
    ) -> None:
        """二值锁存状态机：确认后保持FALL，连续恢复后才回NO_FALL。"""
        now = float(observation.timestamp_s)
        if not memory.is_fall:
            evidence_ok = (
                pose_score >= float(self.state_cfg["min_pose_score"])
                and (
                    height_score >= float(self.state_cfg["min_height_score"])
                    or transient_active
                )
            )
            candidate = (
                observation.data_quality >= float(self.state_cfg["min_data_quality"])
                and fall_score >= float(self.state_cfg["trigger_score"])
                and evidence_ok
            )
            if not candidate:
                memory.candidate_since = None
                return
            if memory.candidate_since is None:
                memory.candidate_since = now
            elif now - memory.candidate_since >= float(self.state_cfg["confirm_duration_s"]):
                memory.is_fall = True
                memory.candidate_since = None
                memory.recover_since = None
            return

        # FALL已经确认后，不因单帧分数下降解除；必须恢复为直立且髋高回升。
        recovered = (
            fall_score <= float(self.state_cfg["release_score"])
            and observation.body_angle_deg is not None
            and observation.body_angle_deg <= float(self.state_cfg["recover_angle_deg"])
            and observation.hip_height_m is not None
            and observation.hip_height_m >= float(self.state_cfg["recover_hip_height_m"])
        )
        if not recovered:
            memory.recover_since = None
            return
        if memory.recover_since is None:
            memory.recover_since = now
        elif now - memory.recover_since >= float(self.state_cfg["recover_duration_s"]):
            memory.is_fall = False
            memory.recover_since = None
            memory.candidate_since = None

    def update(self, observation: FallObservation) -> FallDecision:
        """更新一人五项评分、二值状态并返回当前决策。"""
        memory = self._person(observation.person_id)
        now = float(observation.timestamp_s)
        if memory.last_seen_s and now < memory.last_seen_s:
            raise ValueError("同一人员timestamp_s不能倒退")

        pose_score = self._pose_score(observation.body_angle_deg, observation.bbox_width_height_ratio)
        height_score = self._height_score(memory, now, observation.hip_height_m)
        vertical_velocity = self._vertical_velocity(memory, now, observation.hip_height_m)
        velocity_score = self._velocity_score(vertical_velocity)
        motion_velocity = self._motion_velocity(memory, now, observation.torso_center_3d)
        static_score, static_duration = self._static_score(memory, now, pose_score, height_score, motion_velocity)
        scene_score = self._scene_score(observation.scene_name)

        # 快速下降是短时证据。用有效期锁存后，即使人已落地速度变为0，确认阶段
        # 仍能知道刚刚发生过明显下降，避免“速度证据”和“静止证据”在时间上错开。
        if (
            height_score >= float(self.state_cfg["transient_height_score"])
            or velocity_score >= float(self.state_cfg["transient_velocity_score"])
        ):
            memory.transient_evidence_until = max(
                memory.transient_evidence_until,
                now + float(self.state_cfg["transient_evidence_hold_s"]),
            )
        transient_active = now <= memory.transient_evidence_until

        fall_score = float(
            self.weights["pose"] * pose_score
            + self.weights["height"] * height_score
            + self.weights["velocity"] * velocity_score
            + self.weights["static"] * static_score
            + self.weights["scene"] * scene_score
        )
        # 质量不足时连续缩小风险分，同时状态机还有硬质量门槛。
        quality_floor = float(self.state_cfg["min_data_quality"])
        if observation.data_quality < quality_floor:
            fall_score *= max(0.0, observation.data_quality) / max(quality_floor, 1e-8)
        fall_score = float(np.clip(fall_score, 0.0, 1.0))

        self._update_binary_state(memory, observation, fall_score, pose_score, height_score, transient_active,)

        if observation.hip_height_m is not None:
            memory.height_history.append((now, float(observation.hip_height_m)))
        if observation.torso_center_3d is not None and np.all(np.isfinite(observation.torso_center_3d)):
            memory.center_history.append((now, np.asarray(observation.torso_center_3d, dtype=np.float64)))
        memory.score_history.append((now, fall_score))
        memory.last_seen_s = now

        return FallDecision(
            person_id=int(observation.person_id),
            is_fall=bool(memory.is_fall),
            label="FALL" if memory.is_fall else "NO_FALL",
            fall_score=fall_score,
            pose_score=pose_score,
            height_score=height_score,
            velocity_score=velocity_score,
            static_score=static_score,
            scene_score=scene_score,
            vertical_velocity_mps=vertical_velocity,
            motion_velocity_mps=motion_velocity,
            static_duration_s=static_duration,
            data_quality=float(observation.data_quality),
            transient_evidence_active=transient_active,
        )

    def prune_stale(self, now_s: float) -> Sequence[int]:
        """清理长时间未出现的Track ID，并返回被清理的ID。"""
        stale_after = float(self.state_cfg["stale_person_s"])
        stale_ids = [person_id for person_id, memory in self.people.items() if now_s - memory.last_seen_s > stale_after]
        for person_id in stale_ids:
            del self.people[person_id]
        return stale_ids

    def reset_person(self, person_id: int) -> None:
        """主动清除一个离开画面的Track ID及其全部评分、计时和状态历史。"""
        self.people.pop(int(person_id), None)

def run_self_test(config: dict) -> None:
    """模拟站立->快速跌倒->地面静止->重新站立，并验证床上躺卧不报警。"""
    detector = FallDetector(config)

    def observe(person_id: int, timestamp: float, angle: float, hip_height: float, center_y: float, scene_name: str,) -> FallDecision:
        """构造一帧合成观测并送入状态机。"""
        return detector.update(
            FallObservation(
                person_id=person_id,
                timestamp_s=timestamp,
                body_angle_deg=angle,
                bbox_width_height_ratio=1.0 if angle > 60.0 else 0.35,
                hip_height_m=hip_height,
                torso_center_3d=np.array([0.0, center_y, 2.0], dtype=np.float64),
                scene_name=scene_name,
                data_quality=0.95,
            )
        )

    decision = None
    for timestamp in np.arange(0.0, 1.2, 0.2):
        decision = observe(1, float(timestamp), 10.0, 0.90, 0.0, "unknown")
        assert not decision.is_fall

    observe(1, 1.2, 45.0, 0.65, 0.20, "unknown")
    observe(1, 1.4, 80.0, 0.22, 0.60, "floor")
    for timestamp in np.arange(1.6, 3.2, 0.2):
        decision = observe(1, float(timestamp), 82.0, 0.18, 0.60, "floor")
    assert decision is not None and decision.is_fall

    for timestamp in np.arange(3.2, 6.0, 0.2):
        decision = observe(1, float(timestamp), 10.0, 0.90, 0.0, "unknown")
    assert decision is not None and not decision.is_fall

    # 另一人直接躺在床上：姿态水平但没有低位/下降证据，场景也是低风险。
    for timestamp in np.arange(0.0, 3.0, 0.2):
        decision = observe(2, float(timestamp), 80.0, 0.55, 0.45, "bed")
        assert not decision.is_fall

    print("fall_detector self-test: PASS")
    print("  fall confirm, recovery, normal lying on bed=PASS")

def main() -> None:
    """命令行独立测试入口；默认打开相机，--self-test无需硬件。"""
    parser = argparse.ArgumentParser(description="跌倒评分与二值状态机")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"), help="统一配置文件路径",)
    parser.add_argument("--self-test", action="store_true", help="运行合成时序测试")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    else:
        # 在线测试复用main.py的采集链路，本文件仍只保留评分与二值状态机。
        from main import run_live

        run_live(config, args.config, stage="fall_detector")

if __name__ == "__main__":
    main()

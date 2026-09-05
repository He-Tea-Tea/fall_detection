# -*- coding: utf-8 -*-
"""五维跌倒评分和二值状态机。

本文件不计算人体角度、高度、速度、静止时间或场景关系，只接收其他模块
已经计算完成的五个0～1风险分数：

P：姿态风险，人体越接近水平，分数越高。
H：高度风险，髋部下降越多、离地越低，分数越高。
V：速度风险，髋部向下运动越快，分数越高。
S：静止风险，异常姿态后静止越久，分数越高。
C：场景风险，越像躺在地面而不是床或沙发，分数越高。

完整五维都有效时：
FallScore = 0.30P + 0.25H + 0.20V + 0.15S + 0.10C

某些维度缺失时，不把缺失数据当成正常0分，而是移除对应权重并重新归一化。
例如只有P和C有效：
FallScore = (0.30P + 0.10C) / (0.30 + 0.10)

状态机对外只输出两种结果：
NO_FALL：当前未确认跌倒。
FALL：当前已经确认跌倒。

完整3D路径使用姿态、高度和瞬态证据确认跌倒；Depth或P3D缺失时，可以
按照config.yaml中的降级参数，使用更严格、更慢的2D规则继续判断。
"""

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

# 五维名称、顺序和画面显示字母保持固定。
DIMENSION_NAMES = ("pose", "height", "velocity", "static", "scene")
DIMENSION_LABELS = {
    "pose": "P",
    "height": "H",
    "velocity": "V",
    "static": "S",
    "scene": "C",
}


@dataclass
class FallEvidence:
    """main.py每帧传入的一人五维证据。

    分数为0和数据无效是两种不同情况：

    score=0且valid=True：
        该维度成功完成测量，只是没有发现跌倒风险。

    score=0且valid=False：
        该维度没有可靠数据，这个0只是占位值，不能参与总分计算。
    """

    person_id: int                         # 稳定Track ID
    timestamp_s: float                     # 当前时间，单位秒
    pose_score: float                      # 姿态风险分P
    height_score: float                    # 高度风险分H
    velocity_score: float                  # 下降速度风险分V
    static_score: float                    # 异常后静止风险分S
    scene_score: float                     # 场景风险分C
    body_angle_3d_deg: Optional[float]      # 3D人体角度，用于恢复判断
    hip_height_m: Optional[float]           # 髋部离地高度，用于恢复判断
    data_quality: float                    # 当前有效证据质量，不是跌倒概率

    # 每个valid字段表示对应分数是否有可靠数据。
    pose_valid: bool = True
    height_valid: bool = True
    velocity_valid: bool = True
    static_valid: bool = True
    scene_valid: bool = True

    # 单独记录P中是否包含有效P3D，用于选择普通或2D降级规则。
    pose_3d_valid: bool = True

    # v0.95把H中的“真实历史下降”单独传入；绝对低高度不能再锁存瞬态。
    height_drop_score: float = 0.0
    height_drop_valid: bool = False


@dataclass
class FallDecision:
    """FallDetector每帧对外返回的最终判断。"""

    person_id: int
    is_fall: bool                          # True表示已经确认跌倒
    label: str                             # FALL或NO_FALL
    fall_score: float                      # 有效维度重新加权后的总分
    pose_score: float                      # P，无效时仅为0占位
    height_score: float                    # H，无效时仅为0占位
    velocity_score: float                  # V，无效时仅为0占位
    static_score: float                    # S，无效时仅为0占位
    scene_score: float                     # C，无效时仅为0占位
    data_quality: float                    # 本帧证据质量
    transient_evidence_active: bool        # 近期是否出现过快速下降证据
    available_weight: float                # 本帧有效维度的原始权重总和
    valid_dimensions: Tuple[str, ...]       # 本帧有效维度，如("P","H","C")
    degraded_mode: bool                    # True表示当前没有可靠P3D


@dataclass
class _PersonMemory:
    """一个人的状态机内部记忆，不直接输出给业务层。"""

    history_length: int
    score_history: Deque[Tuple[float, float]] = field(init=False)

    # 候选跌倒开始时间和候选模式。
    candidate_since: Optional[float] = None
    candidate_mode: Optional[str] = None

    # 恢复开始时间和恢复模式。
    recover_since: Optional[float] = None
    recover_mode: Optional[str] = None

    # 真实高度下降分或V出现瞬态高分后，将证据保留到这个时间。
    transient_evidence_until: float = 0.0

    # 对外二值状态。
    is_fall: bool = False

    # 最后一次看到该Track ID的时间。
    last_seen_s: float = 0.0

    def __post_init__(self) -> None:
        """根据配置创建固定长度的总分历史队列。"""
        self.score_history = deque(maxlen=self.history_length)


def load_config(path: str) -> dict:
    """读取统一配置文件config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


class FallDetector:
    """按Track ID维护五维评分和FALL/NO_FALL二值状态。"""

    def __init__(self, config: dict):
        self.cfg = config["fall_detector"]
        self.weights = self.cfg["weights"]
        self.state_cfg = self.cfg["state_machine"]
        self.history_length = int(self.cfg["history_length"])
        self.people: Dict[int, _PersonMemory] = {}

        # 五项权重必须非负，并且完整五维权重总和必须等于1。
        weight_values = [
            float(self.weights[name])
            for name in DIMENSION_NAMES
        ]
        if any(value < 0.0 for value in weight_values):
            raise ValueError("fall_detector.weights不能包含负数")
        if abs(sum(weight_values) - 1.0) > 1e-6:
            raise ValueError("fall_detector.weights总和必须等于1")

    def _person(self, person_id: int) -> _PersonMemory:
        """获取指定人员的状态；第一次出现时自动创建。"""
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _PersonMemory(
                history_length=self.history_length
            )
        return self.people[person_id]

    @staticmethod
    def _bounded(value: float) -> float:
        """将分数限制在0～1，防止上游异常值破坏状态机。"""
        return float(np.clip(value, 0.0, 1.0))

    def _prepare_scores(
        self,
        evidence: FallEvidence,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """整理五维分数，并判断每个维度是否真的有效。"""
        raw_scores = {
            "pose": evidence.pose_score,
            "height": evidence.height_score,
            "velocity": evidence.velocity_score,
            "static": evidence.static_score,
            "scene": evidence.scene_score,
        }
        requested_validity = {
            "pose": evidence.pose_valid,
            "height": evidence.height_valid,
            "velocity": evidence.velocity_valid,
            "static": evidence.static_valid,
            "scene": evidence.scene_valid,
        }

        # 即使上游标记valid=True，NaN和无穷值仍然必须视为无效。
        validity = {
            name: (
                bool(requested_validity[name])
                and bool(np.isfinite(raw_scores[name]))
            )
            for name in DIMENSION_NAMES
        }

        # 无效维度暂时保存为0，但后续加权时不会参与计算。
        scores = {
            name: (
                self._bounded(raw_scores[name])
                if validity[name]
                else 0.0
            )
            for name in DIMENSION_NAMES
        }
        return scores, validity

    def _update_transient_evidence(
        self,
        memory: _PersonMemory,
        scores: Dict[str, float],
        validity: Dict[str, bool],
        evidence: FallEvidence,
        now: float,
    ) -> bool:
        """保存短暂出现的高度下降或快速下降证据。

        跌倒过程通常分成几个阶段：
        1. 人体快速向下运动，此时V升高；
        2. 髋部高度降低，此时H升高；
        3. 人体最终水平并保持静止，此时P和S升高。

        真实下降分和V可能只持续很短。如果不保存瞬态证据，等到P、S升高时，
        下降或速度可能已经降低，导致几个证据无法在同一帧组合。
        """
        # 这里必须使用height.py单独输出的drop_score，不能使用聚合后的H。
        # 因此“当前髋高很低”只能作为当前高度证据，不会伪装成刚刚跌落。
        height_triggered = (
            bool(evidence.height_drop_valid)
            and np.isfinite(evidence.height_drop_score)
            and float(evidence.height_drop_score)
            >= float(self.state_cfg["transient_height_score"])
        )
        velocity_triggered = (
            validity["velocity"]
            and scores["velocity"]
            >= float(self.state_cfg["transient_velocity_score"])
        )

        if height_triggered or velocity_triggered:
            hold_seconds = float(
                self.state_cfg["transient_evidence_hold_s"]
            )
            memory.transient_evidence_until = max(
                memory.transient_evidence_until,
                now + hold_seconds,
            )

        return (
            memory.transient_evidence_until > 0.0
            and now <= memory.transient_evidence_until
        )

    def _calculate_fall_score(
        self,
        scores: Dict[str, float],
        validity: Dict[str, bool],
        data_quality: float,
    ) -> Tuple[float, float]:
        """只使用有效维度计算总分，并返回有效权重总和。"""
        available_weight = sum(
            float(self.weights[name])
            for name in DIMENSION_NAMES
            if validity[name]
        )
        weighted_sum = sum(
            float(self.weights[name]) * scores[name]
            for name in DIMENSION_NAMES
            if validity[name]
        )

        # 所有维度都无效时，没有证据，总分只能为0。
        if available_weight <= 0.0:
            fall_score = 0.0
        else:
            # 缺失维度被移除后，按剩余有效权重重新归一化。
            fall_score = weighted_sum / available_weight

        # 质量低于最低阈值时，进一步衰减总分。
        quality_floor = float(
            self.state_cfg["min_data_quality"]
        )
        if data_quality < quality_floor:
            quality_ratio = max(
                0.0,
                float(data_quality),
            ) / max(quality_floor, 1e-8)
            fall_score *= quality_ratio

        return self._bounded(fall_score), float(available_weight)

    def _update_no_fall_state(
        self,
        memory: _PersonMemory,
        evidence: FallEvidence,
        fall_score: float,
        transient_active: bool,
        validity: Dict[str, bool],
        available_weight: float,
    ) -> None:
        """当前为NO_FALL时，判断是否应该确认成FALL。"""
        now = float(evidence.timestamp_s)

        # 两条确认路径都要求基础数据质量合格。
        quality_ok = (
            evidence.data_quality
            >= float(self.state_cfg["min_data_quality"])
        )

        # 姿态是核心条件，无有效P时不能确认新的跌倒。
        pose_ok = (
            validity["pose"]
            and evidence.pose_score
            >= float(self.state_cfg["min_pose_score"])
        )

        # 完整路径要求当前H足够高，或者近期曾出现过H/V瞬态证据。
        height_support = (
            validity["height"]
            and evidence.height_score
            >= float(self.state_cfg["min_height_score"])
        ) or transient_active

        normal_candidate = (
            quality_ok
            and pose_ok
            and height_support
            and fall_score
            >= float(self.state_cfg["trigger_score"])
        )

        # P3D和H都缺失时，可以进入纯2D降级判断。
        # 降级模式要求更高P、更高总分和更长确认时间，减少误报。
        degraded_candidate = (
            bool(self.state_cfg.get("allow_2d_only_fall", True))
            and quality_ok
            and not evidence.pose_3d_valid
            and not validity["height"]
            and validity["pose"]
            and evidence.pose_score
            >= float(
                self.state_cfg.get(
                    "degraded_min_pose_score",
                    0.85,
                )
            )
            and available_weight
            >= float(
                self.state_cfg.get(
                    "degraded_min_available_weight",
                    0.30,
                )
            )
            and fall_score
            >= float(
                self.state_cfg.get(
                    "degraded_trigger_score",
                    0.80,
                )
            )
        )

        if normal_candidate:
            candidate_mode = "normal"
        elif degraded_candidate:
            candidate_mode = "degraded_2d"
        else:
            candidate_mode = None

        # 当前证据不再满足条件，候选计时必须重新开始。
        if candidate_mode is None:
            memory.candidate_since = None
            memory.candidate_mode = None
            return

        # 普通模式和2D降级模式不能共用同一段确认时间。
        if (
            memory.candidate_since is None
            or memory.candidate_mode != candidate_mode
        ):
            memory.candidate_since = now
            memory.candidate_mode = candidate_mode
            return

        if candidate_mode == "normal":
            confirm_duration = float(
                self.state_cfg["confirm_duration_s"]
            )
        else:
            confirm_duration = float(
                self.state_cfg.get(
                    "degraded_confirm_duration_s",
                    2.0,
                )
            )

        # 高风险证据连续保持足够时间后，才真正切换到FALL。
        if now - memory.candidate_since >= confirm_duration:
            memory.is_fall = True
            memory.candidate_since = None
            memory.candidate_mode = None
            memory.recover_since = None
            memory.recover_mode = None

    def _update_fall_state(
        self,
        memory: _PersonMemory,
        evidence: FallEvidence,
        fall_score: float,
        validity: Dict[str, bool],
    ) -> None:
        """当前为FALL时，判断人员是否已经恢复。"""
        now = float(evidence.timestamp_s)

        # 完整3D恢复必须同时满足：
        # 1. 总风险降低；
        # 2. 3D人体角度恢复为接近竖直；
        # 3. 髋部重新升高。
        normal_recovered = (
            fall_score <= float(self.state_cfg["release_score"])
            and evidence.pose_3d_valid
            and evidence.body_angle_3d_deg is not None
            and evidence.body_angle_3d_deg
            <= float(self.state_cfg["recover_angle_deg"])
            and evidence.hip_height_m is not None
            and evidence.hip_height_m
            >= float(self.state_cfg["recover_hip_height_m"])
        )

        # 如果报警后3D仍然缺失，可以使用持续正常的P2D恢复。
        # 该路径使用更长时间，防止2D关键点短暂抖动解除真实报警。
        degraded_recovered = (
            bool(self.state_cfg.get("allow_2d_only_fall", True))
            and not evidence.pose_3d_valid
            and validity["pose"]
            and evidence.pose_score
            <= float(
                self.state_cfg.get(
                    "degraded_recover_pose_score",
                    0.25,
                )
            )
            and fall_score
            <= float(self.state_cfg["release_score"])
        )

        if normal_recovered:
            recover_mode = "normal"
        elif degraded_recovered:
            recover_mode = "degraded_2d"
        else:
            recover_mode = None

        # 恢复条件中断时重新计时。
        # 如果所有数据丢失，recover_mode也是None，因此不会自动解除FALL。
        if recover_mode is None:
            memory.recover_since = None
            memory.recover_mode = None
            return

        if (
            memory.recover_since is None
            or memory.recover_mode != recover_mode
        ):
            memory.recover_since = now
            memory.recover_mode = recover_mode
            return

        if recover_mode == "normal":
            recover_duration = float(
                self.state_cfg["recover_duration_s"]
            )
        else:
            recover_duration = float(
                self.state_cfg.get(
                    "degraded_recover_duration_s",
                    3.0,
                )
            )

        # 恢复证据连续保持足够时间后，才解除报警。
        if now - memory.recover_since >= recover_duration:
            memory.is_fall = False
            memory.recover_since = None
            memory.recover_mode = None
            memory.candidate_since = None
            memory.candidate_mode = None

    def _update_binary_state(
        self,
        memory: _PersonMemory,
        evidence: FallEvidence,
        fall_score: float,
        transient_active: bool,
        validity: Dict[str, bool],
        available_weight: float,
    ) -> None:
        """根据当前状态选择跌倒确认或恢复逻辑。"""
        if memory.is_fall:
            self._update_fall_state(
                memory,
                evidence,
                fall_score,
                validity,
            )
        else:
            self._update_no_fall_state(
                memory,
                evidence,
                fall_score,
                transient_active,
                validity,
                available_weight,
            )

    def update(self, evidence: FallEvidence) -> FallDecision:
        """接收一帧五维证据，更新状态机并返回最终判断。"""
        person_id = int(evidence.person_id)
        now = float(evidence.timestamp_s)
        memory = self._person(person_id)

        # 同一个人的时间不能倒退，否则所有持续时间计算都会失真。
        if memory.last_seen_s and now < memory.last_seen_s:
            raise ValueError("同一人员timestamp_s不能倒退")

        scores, validity = self._prepare_scores(evidence)

        transient_active = self._update_transient_evidence(
            memory,
            scores,
            validity,
            evidence,
            now,
        )
        fall_score, available_weight = self._calculate_fall_score(
            scores,
            validity,
            evidence.data_quality,
        )

        # 使用限制到0～1后的分数更新状态机，避免上下游解释不一致。
        normalized_evidence = FallEvidence(
            person_id=person_id,
            timestamp_s=now,
            pose_score=scores["pose"],
            height_score=scores["height"],
            velocity_score=scores["velocity"],
            static_score=scores["static"],
            scene_score=scores["scene"],
            body_angle_3d_deg=evidence.body_angle_3d_deg,
            hip_height_m=evidence.hip_height_m,
            data_quality=float(evidence.data_quality),
            pose_valid=validity["pose"],
            height_valid=validity["height"],
            velocity_valid=validity["velocity"],
            static_valid=validity["static"],
            scene_valid=validity["scene"],
            pose_3d_valid=bool(evidence.pose_3d_valid),
            height_drop_score=self._bounded(evidence.height_drop_score),
            height_drop_valid=bool(evidence.height_drop_valid),
        )

        self._update_binary_state(
            memory,
            normalized_evidence,
            fall_score,
            transient_active,
            validity,
            available_weight,
        )

        memory.score_history.append((now, fall_score))
        memory.last_seen_s = now

        valid_dimensions = tuple(
            DIMENSION_LABELS[name]
            for name in DIMENSION_NAMES
            if validity[name]
        )

        return FallDecision(
            person_id=person_id,
            is_fall=bool(memory.is_fall),
            label="FALL" if memory.is_fall else "NO_FALL",
            fall_score=fall_score,
            pose_score=scores["pose"],
            height_score=scores["height"],
            velocity_score=scores["velocity"],
            static_score=scores["static"],
            scene_score=scores["scene"],
            data_quality=float(evidence.data_quality),
            transient_evidence_active=transient_active,
            available_weight=available_weight,
            valid_dimensions=valid_dimensions,
            degraded_mode=not bool(evidence.pose_3d_valid),
        )

    def prune_stale(self, now_s: float) -> Sequence[int]:
        """删除长时间未出现的Track ID，避免旧状态长期占用内存。"""
        now = float(now_s)
        stale_after = float(
            self.state_cfg["stale_person_s"]
        )
        stale_ids = [
            person_id
            for person_id, memory in self.people.items()
            if now - memory.last_seen_s > stale_after
        ]

        for person_id in stale_ids:
            del self.people[person_id]

        return stale_ids

    def reset_person(self, person_id: int) -> None:
        """立即清除指定人员的评分历史和状态。"""
        self.people.pop(int(person_id), None)

    def invalidate_transient_measurements(self, person_id: int) -> None:
        """地面切换后清除候选与瞬态计时，但保留已经确认的FALL状态。"""
        memory = self.people.get(int(person_id))
        if memory is None:
            return
        memory.candidate_since = None
        memory.candidate_mode = None
        memory.transient_evidence_until = 0.0
        memory.score_history.clear()


def run_self_test(config: dict) -> None:
    """验证完整五维、2D降级、报警恢复和床上躺卧抑制。"""
    detector = FallDetector(config)

    # H=1可能只表示当前位置低；没有真实历史下降时不能锁存瞬态证据。
    absolute_only = detector.update(
        FallEvidence(
            person_id=90,
            timestamp_s=0.0,
            pose_score=0.2,
            height_score=1.0,
            velocity_score=0.0,
            static_score=0.0,
            scene_score=0.5,
            body_angle_3d_deg=20.0,
            hip_height_m=0.20,
            data_quality=0.95,
            height_drop_score=0.0,
            height_drop_valid=False,
        )
    )
    assert not absolute_only.transient_evidence_active

    def feed_full_evidence(
        person_id: int,
        now: float,
        pose: float,
        height: float,
        velocity: float,
        static: float,
        scene: float,
        angle_3d: Optional[float],
        hip_height: Optional[float],
    ) -> FallDecision:
        """向状态机输入五维全部有效的一帧测试数据。"""
        return detector.update(
            FallEvidence(
                person_id=person_id,
                timestamp_s=now,
                pose_score=pose,
                height_score=height,
                velocity_score=velocity,
                static_score=static,
                scene_score=scene,
                body_angle_3d_deg=angle_3d,
                hip_height_m=hip_height,
                data_quality=0.95,
            )
        )

    # 测试1：正常站立不应报警。
    decision = None
    for now in np.arange(0.0, 1.0, 0.2):
        decision = feed_full_evidence(
            person_id=1,
            now=float(now),
            pose=0.0,
            height=0.0,
            velocity=0.0,
            static=0.0,
            scene=0.5,
            angle_3d=10.0,
            hip_height=0.90,
        )
        assert not decision.is_fall

    # 测试2：完整跌倒证据持续超过确认时间后，应切换到FALL。
    for now in np.arange(1.0, 2.6, 0.2):
        decision = feed_full_evidence(
            person_id=1,
            now=float(now),
            pose=1.0,
            height=1.0,
            velocity=0.8 if now < 1.6 else 0.0,
            static=1.0,
            scene=1.0,
            angle_3d=82.0,
            hip_height=0.18,
        )
    assert decision is not None and decision.is_fall

    # 测试3：恢复直立且髋高升高，持续超过恢复时间后解除报警。
    for now in np.arange(2.6, 5.2, 0.2):
        decision = feed_full_evidence(
            person_id=1,
            now=float(now),
            pose=0.0,
            height=0.0,
            velocity=0.0,
            static=0.0,
            scene=0.5,
            angle_3d=10.0,
            hip_height=0.90,
        )
    assert decision is not None and not decision.is_fall

    # 测试4：床上水平躺卧虽然P和S较高，但C和H较低，不应报警。
    for now in np.arange(0.0, 3.0, 0.2):
        decision = feed_full_evidence(
            person_id=2,
            now=float(now),
            pose=1.0,
            height=0.0,
            velocity=0.0,
            static=1.0,
            scene=0.0,
            angle_3d=80.0,
            hip_height=0.55,
        )
        assert not decision.is_fall

    # 测试5：Depth完全缺失，只有P2D和unknown场景有效。
    for now in np.arange(0.0, 2.6, 0.2):
        decision = detector.update(
            FallEvidence(
                person_id=3,
                timestamp_s=float(now),
                pose_score=1.0,
                height_score=0.0,
                velocity_score=0.0,
                static_score=0.0,
                scene_score=0.5,
                body_angle_3d_deg=None,
                hip_height_m=None,
                data_quality=0.95,
                pose_valid=True,
                height_valid=False,
                velocity_valid=False,
                static_valid=False,
                scene_valid=True,
                pose_3d_valid=False,
            )
        )

    assert decision is not None
    assert decision.is_fall
    assert decision.degraded_mode
    assert decision.valid_dimensions == ("P", "C")

    # 只有P和C有效时，总分为(0.30×1+0.10×0.5)/0.40=0.875。
    assert abs(decision.fall_score - 0.875) < 1e-6

    # 测试6：3D仍未恢复，但P2D持续恢复正常，应通过降级规则解除报警。
    for now in np.arange(2.6, 6.2, 0.2):
        decision = detector.update(
            FallEvidence(
                person_id=3,
                timestamp_s=float(now),
                pose_score=0.0,
                height_score=0.0,
                velocity_score=0.0,
                static_score=0.0,
                scene_score=0.5,
                body_angle_3d_deg=None,
                hip_height_m=None,
                data_quality=0.95,
                pose_valid=True,
                height_valid=False,
                velocity_valid=False,
                static_valid=False,
                scene_valid=True,
                pose_3d_valid=False,
            )
        )

    assert not decision.is_fall

    print("fall_detector self-test: PASS")
    print(
        "  drop-only transient, full five-score path, missing-3D fallback, "
        "recovery, bed suppression=PASS"
    )


def main() -> None:
    """独立运行入口：--self-test测试，默认调用main.py打开相机。"""
    parser = argparse.ArgumentParser(
        description="五项加权与二值跌倒状态机"
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
        # 相机只允许由main.py统一打开，避免多个文件重复占用相机。
        from main import run_live

        run_live(
            config,
            args.config,
            stage="fall_detector",
        )


if __name__ == "__main__":
    main()

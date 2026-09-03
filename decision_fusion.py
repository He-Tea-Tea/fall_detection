# -*- coding: utf-8 -*-
"""本地跌倒结果与AI复核结果融合模块。

融合规则保持保守和可解释：
1. 本地状态机已经确认FALL时立即输出FALL，AI不能否决；
2. 本地仍为NO_FALL但风险达到最低门槛时，高置信度AI可以辅助确认；
3. 文本模型只读取本地结构化数据，触发门槛高于真正读取图片的视觉模型；
4. AI超时、报错、不确定或结果过期时，最终结果完全退回本地判断；
5. AI辅助确认后保持一段时间，避免网络结果只出现一帧就消失。

本文件不调用网络，也不计算P/H/V/S/C，只负责组合两个已经完成的结果。
"""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import yaml

from ai_verifier import (
    AI_FALL,
    AIVerificationResult,
)
from fall_detector import FallDecision


@dataclass
class FusionDecision:
    """系统最终二值判断，同时保留本地和AI来源便于追溯。"""

    person_id: int
    is_fall: bool
    label: str
    source: str
    local_is_fall: bool
    local_fall_score: float
    ai_verdict: str
    ai_confidence: float
    ai_event_id: str
    reason: str


@dataclass
class _FusionMemory:
    """一个人的AI辅助FALL保持时间和最后事件编号。"""

    ai_fall_until_s: float = 0.0
    last_ai_event_id: str = ""


class DecisionFusion:
    """按Track ID融合本地二值状态和最新AI复核。"""

    def __init__(self, config: dict):
        self.cfg = config["ai"]["fusion"]
        self.people: Dict[int, _FusionMemory] = {}

    def _person(self, person_id: int) -> _FusionMemory:
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _FusionMemory()
        return self.people[person_id]

    def _ai_can_confirm(
        self,
        local: FallDecision,
        ai_result: Optional[AIVerificationResult],
    ) -> bool:
        """检查AI结果、置信度和本地最低证据是否同时满足。"""
        if ai_result is None or not ai_result.usable:
            return False
        if ai_result.verdict != AI_FALL:
            return False
        if ai_result.confidence < float(self.cfg["min_ai_fall_confidence"]):
            return False

        # 视觉模型看到了独立图像证据，因此允许较低的本地总分门槛。
        # 当前文本模型只复核P/H/V/S/C，使用更高门槛避免重复证据被放大。
        score_key = (
            "vision_min_local_score"
            if ai_result.includes_images
            else "text_min_local_score"
        )
        return bool(
            max(local.fall_score, ai_result.peak_local_score)
            >= float(self.cfg[score_key])
            and max(local.pose_score, ai_result.peak_pose_score)
            >= float(self.cfg["min_local_pose_score"])
            and max(local.data_quality, ai_result.peak_data_quality)
            >= float(self.cfg["min_local_data_quality"])
        )

    def update(
        self,
        local: FallDecision,
        ai_result: Optional[AIVerificationResult],
        timestamp_s: float,
    ) -> FusionDecision:
        """输出系统最终FALL/NO_FALL；任何AI异常都不会阻塞本地结果。"""
        person_id = int(local.person_id)
        now = float(timestamp_s)
        memory = self._person(person_id)

        # 同一个AI事件只处理一次，防止每帧反复延长保持时间。
        if (
            ai_result is not None
            and ai_result.event_id != memory.last_ai_event_id
        ):
            memory.last_ai_event_id = ai_result.event_id
            if self._ai_can_confirm(local, ai_result):
                memory.ai_fall_until_s = max(
                    memory.ai_fall_until_s,
                    now + float(self.cfg["positive_hold_s"]),
                )

        ai_hold_active = bool(
            memory.ai_fall_until_s > 0.0
            and now <= memory.ai_fall_until_s
        )
        ai_verdict = ai_result.verdict if ai_result is not None else "NONE"
        ai_confidence = ai_result.confidence if ai_result is not None else 0.0
        ai_event_id = ai_result.event_id if ai_result is not None else ""

        # 本地FALL优先级最高，AI无权将它改为NO_FALL。
        if local.is_fall:
            source = "LOCAL+AI" if ai_verdict == AI_FALL else "LOCAL"
            reason = "本地五维状态机已经确认跌倒"
            final_fall = True
        elif ai_hold_active:
            source = "AI_ASSISTED"
            reason = "本地存在疑似证据，并获得高置信度AI复核支持"
            final_fall = True
        else:
            source = "LOCAL_ONLY"
            reason = "本地状态机未确认跌倒"
            final_fall = False

        return FusionDecision(
            person_id=person_id,
            is_fall=final_fall,
            label="FALL" if final_fall else "NO_FALL",
            source=source,
            local_is_fall=bool(local.is_fall),
            local_fall_score=float(local.fall_score),
            ai_verdict=ai_verdict,
            ai_confidence=float(ai_confidence),
            ai_event_id=ai_event_id,
            reason=reason,
        )

    def reset_person(self, person_id: int) -> None:
        """清除离场Track ID的AI辅助状态。"""
        self.people.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    """验证本地优先、文本AI辅助、低置信度忽略和保持时间。"""
    fusion = DecisionFusion(config)

    def local_decision(is_fall: bool, score: float) -> FallDecision:
        return FallDecision(
            person_id=1,
            is_fall=is_fall,
            label="FALL" if is_fall else "NO_FALL",
            fall_score=score,
            pose_score=0.90,
            height_score=0.70,
            velocity_score=0.50,
            static_score=0.60,
            scene_score=1.00,
            data_quality=0.95,
            transient_evidence_active=True,
            available_weight=1.00,
            valid_dimensions=("P", "H", "V", "S", "C"),
            degraded_mode=False,
        )

    ai_fall = AIVerificationResult(
        event_id="event-1",
        person_id=1,
        success=True,
        verdict=AI_FALL,
        confidence=0.95,
        risk_level="high",
        reason="连续异常",
        observations=("P较高",),
        requested_s=0.0,
        completed_s=0.1,
        latency_s=0.1,
        model="test-model",
        includes_images=False,
        peak_local_score=0.70,
        peak_pose_score=0.90,
        peak_data_quality=0.95,
    )

    local_only = fusion.update(local_decision(False, 0.20), None, 0.0)
    assert not local_only.is_fall

    low_risk_fusion = DecisionFusion(config)
    low_risk_ai = replace(
        ai_fall,
        event_id="event-low-risk",
        peak_local_score=0.20,
        peak_pose_score=0.20,
    )
    ignored_ai = low_risk_fusion.update(
        local_decision(False, 0.20),
        low_risk_ai,
        0.5,
    )
    assert not ignored_ai.is_fall

    ai_assisted = fusion.update(local_decision(False, 0.70), ai_fall, 1.0)
    assert ai_assisted.is_fall and ai_assisted.source == "AI_ASSISTED"

    held = fusion.update(local_decision(False, 0.10), ai_fall, 2.0)
    assert held.is_fall

    local_fall = fusion.update(local_decision(True, 0.90), None, 3.0)
    assert local_fall.is_fall and local_fall.source == "LOCAL"
    print("decision_fusion self-test: PASS")
    print("  local priority, AI-assisted confirmation, positive hold=PASS")


def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """--self-test独立验证融合规则；默认复用main.py打开AI窗口。"""
    parser = argparse.ArgumentParser(description="本地与AI跌倒结果融合")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
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
        from main import run_live

        run_live(config, args.config, stage="ai")


if __name__ == "__main__":
    main()

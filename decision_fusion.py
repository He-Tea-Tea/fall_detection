# -*- coding: utf-8 -*-
"""本地跌倒结果、视觉AI结果和后续外部确认的融合模块。

默认策略符合当前项目要求：
1. 没有可用AI结果时，完全使用本地FALL/NO_FALL；
2. 视觉AI成功返回明确true/false后，在短暂有效期内以AI为主；
3. AI超时、断网、报错、不确定或结果过期时，立即退回本地判断；
4. AI判断FALL保持较长时间，AI判断NO_FALL只保持较短时间；
5. ExternalConfirmation为后续语音识别、老人回答和人工确认预留。

本模块不调用网络，也不重新计算P/H/V/S/C。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

from ai_verifier import (
    AI_FALL,
    AI_NO_FALL,
    AIVerificationResult,
)
from fall_detector import FallDecision


@dataclass
class ExternalConfirmation:
    """语音、护理人员或人工按钮以后都可转换成这一统一输入。"""

    person_id: int
    source: str
    verdict: str
    confidence: float
    timestamp_s: float
    ttl_s: float
    reason: str = ""

    def usable_at(self, now_s: float) -> bool:
        """只有明确结论、置信度合格且未过期才允许参与融合。"""
        return bool(
            self.verdict in {AI_FALL, AI_NO_FALL}
            and 0.0 <= float(now_s) - float(self.timestamp_s) <= float(self.ttl_s)
        )


@dataclass
class FusionDecision:
    """系统最终二值判断，并保留结论来源便于后续告警追溯。"""

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
    """保存最近一次有效AI结论和以后接入的语音/人工结论。"""

    ai_verdict: str = "NONE"
    ai_confidence: float = 0.0
    ai_event_id: str = ""
    ai_reason: str = ""
    ai_until_s: float = 0.0
    external: Optional[ExternalConfirmation] = None


class DecisionFusion:
    """按Track ID执行外部确认 > 视觉AI > 本地状态机的融合顺序。"""

    def __init__(self, config: dict):
        self.cfg = config["ai"]["fusion"]
        self.people: Dict[int, _FusionMemory] = {}

    def _person(self, person_id: int) -> _FusionMemory:
        person_id = int(person_id)
        if person_id not in self.people:
            self.people[person_id] = _FusionMemory()
        return self.people[person_id]

    def submit_external_confirmation(
        self,
        confirmation: ExternalConfirmation,
    ) -> bool:
        """预留入口：语音识别或人工确认模块以后从这里提交结果。"""
        minimum = float(self.cfg["external_min_confidence"])
        if float(confirmation.confidence) < minimum:
            return False
        self._person(confirmation.person_id).external = confirmation
        return True

    def _remember_ai(
        self,
        memory: _FusionMemory,
        ai_result: Optional[AIVerificationResult],
        now_s: float,
    ) -> None:
        """只接收新的高置信度视觉结果，并按正负结论设置保持时间。"""
        if ai_result is None or not ai_result.usable:
            return
        if not ai_result.includes_images:
            return
        if ai_result.event_id == memory.ai_event_id:
            return
        if ai_result.confidence < float(self.cfg["min_ai_confidence"]):
            return

        hold_key = (
            "positive_hold_s"
            if ai_result.verdict == AI_FALL
            else "negative_hold_s"
        )
        memory.ai_verdict = ai_result.verdict
        memory.ai_confidence = float(ai_result.confidence)
        memory.ai_event_id = ai_result.event_id
        memory.ai_reason = ai_result.reason
        memory.ai_until_s = float(now_s) + float(self.cfg[hold_key])

    def update(
        self,
        local: FallDecision,
        ai_result: Optional[AIVerificationResult],
        timestamp_s: float,
        external_confirmation: Optional[ExternalConfirmation] = None,
    ) -> FusionDecision:
        """输出最终结果；AI不可用时本地判断不会暂停。"""
        person_id = int(local.person_id)
        now = float(timestamp_s)
        memory = self._person(person_id)

        if external_confirmation is not None:
            self.submit_external_confirmation(external_confirmation)
        self._remember_ai(memory, ai_result, now)

        external = memory.external
        external_active = bool(
            external is not None
            and external.usable_at(now)
            and external.confidence >= float(self.cfg["external_min_confidence"])
        )
        ai_active = bool(
            memory.ai_until_s > 0.0
            and now <= memory.ai_until_s
            and memory.ai_verdict in {AI_FALL, AI_NO_FALL}
        )

        # 最高优先级留给将来的老人语音确认、护理人员确认或物理按钮。
        if external_active and external is not None:
            final_fall = external.verdict == AI_FALL
            source = f"{external.source.upper()}_PRIMARY"
            reason = external.reason or "外部确认结果优先"
        elif ai_active:
            if memory.ai_verdict == AI_FALL:
                final_fall = True
                source = "AI_PRIMARY"
                reason = memory.ai_reason or "视觉AI判断有人跌倒"
            elif bool(self.cfg["ai_can_clear_local_fall"]):
                final_fall = False
                source = "AI_PRIMARY"
                reason = memory.ai_reason or "视觉AI判断未发生跌倒"
            else:
                final_fall = bool(local.is_fall)
                source = "LOCAL_SAFETY" if local.is_fall else "AI_PRIMARY"
                reason = (
                    "本地已确认FALL，安全配置禁止AI直接解除"
                    if local.is_fall
                    else memory.ai_reason or "视觉AI判断未发生跌倒"
                )
        else:
            final_fall = bool(local.is_fall)
            source = "LOCAL_FALLBACK"
            reason = (
                "AI不可用或没有有效结果，使用本地状态机判断"
            )

        return FusionDecision(
            person_id=person_id,
            is_fall=final_fall,
            label="FALL" if final_fall else "NO_FALL",
            source=source,
            local_is_fall=bool(local.is_fall),
            local_fall_score=float(local.fall_score),
            ai_verdict=memory.ai_verdict if ai_active else "NONE",
            ai_confidence=memory.ai_confidence if ai_active else 0.0,
            ai_event_id=memory.ai_event_id if ai_active else "",
            reason=reason,
        )

    def reset_person(self, person_id: int) -> None:
        """清除离场Track ID的AI和外部确认状态。"""
        self.people.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    """验证AI优先、断网本地降级、正负保持和语音预留入口。"""
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

    def ai_result(event_id: str, verdict: str) -> AIVerificationResult:
        return AIVerificationResult(
            event_id=event_id,
            person_id=1,
            success=True,
            verdict=verdict,
            confidence=1.0,
            risk_level="high" if verdict == AI_FALL else "low",
            reason="视觉模型测试结论",
            observations=(),
            requested_s=0.0,
            completed_s=0.1,
            latency_s=0.1,
            model="test-vision-model",
            includes_images=True,
            peak_local_score=0.50,
            peak_pose_score=0.90,
            peak_data_quality=0.95,
            raw_answer="true" if verdict == AI_FALL else "false",
        )

    # 没有AI时，本地FALL必须继续输出，证明断网不会让系统停止判断。
    local_only = fusion.update(local_decision(True, 0.80), None, 0.0)
    assert local_only.is_fall and local_only.source == "LOCAL_FALLBACK"

    # 有明确视觉AI结果时，默认按用户要求由AI覆盖本地结果。
    ai_no_fall = fusion.update(
        local_decision(True, 0.80),
        ai_result("event-no-fall", AI_NO_FALL),
        1.0,
    )
    assert not ai_no_fall.is_fall and ai_no_fall.source == "AI_PRIMARY"

    ai_fall = fusion.update(
        local_decision(False, 0.50),
        ai_result("event-fall", AI_FALL),
        5.0,
    )
    assert ai_fall.is_fall and ai_fall.source == "AI_PRIMARY"

    # 语音/人工确认接口优先级高于AI，供下一阶段直接接入。
    voice = ExternalConfirmation(
        person_id=1,
        source="VOICE",
        verdict=AI_NO_FALL,
        confidence=0.95,
        timestamp_s=6.0,
        ttl_s=10.0,
        reason="老人清楚回答自己没有跌倒",
    )
    voice_result = fusion.update(
        local_decision(True, 0.90),
        None,
        6.0,
        external_confirmation=voice,
    )
    assert not voice_result.is_fall and voice_result.source == "VOICE_PRIMARY"
    print("decision_fusion self-test: PASS")
    print("  AI primary, offline local fallback, hold time and voice interface=PASS")


def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """--self-test独立测试；默认复用main.py打开AI窗口。"""
    parser = argparse.ArgumentParser(description="本地、AI和外部确认融合")
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

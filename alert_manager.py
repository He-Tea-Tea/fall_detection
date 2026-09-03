# -*- coding: utf-8 -*-
"""跌倒告警接口管理器。

当前版本只提供控制台告警，但已经预留统一handler接口。后续接入机器人停车、
云台转头、语音询问、手机通知或护理平台时，只需注册新的handler，不需要修改
本地五维检测、AI调用或融合逻辑。

告警按状态变化触发：NO_FALL变为FALL时发送一次跌倒事件，FALL恢复为
NO_FALL时发送一次恢复事件，不会因为主循环每帧运行而重复发送。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from ai_verifier import AIVerificationResult
from decision_fusion import FusionDecision


@dataclass
class AlertEvent:
    """提供给机器人、语音和远程通知模块的统一事件数据。"""

    event_type: str
    person_id: int
    timestamp_s: float
    final_label: str
    source: str
    local_fall_score: float
    ai_verdict: str
    ai_confidence: float
    reason: str


AlertHandler = Callable[[AlertEvent], None]


class AlertManager:
    """检测最终状态变化，并把事件分发给所有已注册处理器。"""

    def __init__(self, config: dict):
        self.cfg = config["alerts"]
        self.enabled = bool(self.cfg["enabled"])
        self.last_fall_state: Dict[int, bool] = {}
        self.handlers: List[AlertHandler] = []
        if bool(self.cfg["console_enabled"]):
            self.register_handler(self._console_handler)

    def register_handler(self, handler: AlertHandler) -> None:
        """注册一个事件处理函数，后续可用于语音、机器人动作或远程通知。"""
        if not callable(handler):
            raise TypeError("alert handler必须可以被调用")
        self.handlers.append(handler)

    @staticmethod
    def _console_handler(event: AlertEvent) -> None:
        """当前默认处理器：在终端打印状态变化，不执行外部动作。"""
        if event.event_type == "FALL_CONFIRMED":
            print(
                f"[ALERT] ID={event.person_id} FALL "
                f"source={event.source} local={event.local_fall_score:.2f} "
                f"AI={event.ai_verdict}:{event.ai_confidence:.2f}"
            )
        else:
            print(f"[RECOVERED] ID={event.person_id} NO_FALL")

    def update(
        self,
        decision: FusionDecision,
        timestamp_s: float,
        ai_result: Optional[AIVerificationResult] = None,
    ) -> Optional[AlertEvent]:
        """状态改变时生成事件；状态没有变化时返回None。"""
        person_id = int(decision.person_id)
        previous = self.last_fall_state.get(person_id, False)
        current = bool(decision.is_fall)
        self.last_fall_state[person_id] = current
        if not self.enabled or current == previous:
            return None

        event = AlertEvent(
            event_type="FALL_CONFIRMED" if current else "FALL_RECOVERED",
            person_id=person_id,
            timestamp_s=float(timestamp_s),
            final_label=decision.label,
            source=decision.source,
            local_fall_score=float(decision.local_fall_score),
            ai_verdict=(
                ai_result.verdict
                if ai_result is not None
                else "NONE"
            ),
            ai_confidence=(
                float(ai_result.confidence)
                if ai_result is not None
                else 0.0
            ),
            reason=decision.reason,
        )

        # 单个外部告警接口异常不能破坏相机和跌倒检测主循环。
        for handler in tuple(self.handlers):
            try:
                handler(event)
            except Exception as error:
                print(f"警告：告警处理器执行失败：{error}")
        return event

    def reset_person(self, person_id: int) -> None:
        """人员离场时清除状态；不会把离场自动当成恢复事件。"""
        self.last_fall_state.pop(int(person_id), None)


def run_self_test(config: dict) -> None:
    """验证只在FALL和恢复状态变化时各产生一次事件。"""
    test_config = {**config, "alerts": dict(config["alerts"])}
    test_config["alerts"]["enabled"] = True
    test_config["alerts"]["console_enabled"] = False
    manager = AlertManager(test_config)
    received: List[AlertEvent] = []
    manager.register_handler(received.append)

    def decision(is_fall: bool) -> FusionDecision:
        return FusionDecision(
            person_id=1,
            is_fall=is_fall,
            label="FALL" if is_fall else "NO_FALL",
            source="LOCAL",
            local_is_fall=is_fall,
            local_fall_score=0.90 if is_fall else 0.10,
            ai_verdict="NONE",
            ai_confidence=0.0,
            ai_event_id="",
            reason="test",
        )

    assert manager.update(decision(False), 0.0) is None
    assert manager.update(decision(True), 1.0) is not None
    assert manager.update(decision(True), 2.0) is None
    assert manager.update(decision(False), 3.0) is not None
    assert [event.event_type for event in received] == [
        "FALL_CONFIRMED",
        "FALL_RECOVERED",
    ]
    print("alert_manager self-test: PASS")
    print("  fall transition, duplicate suppression, recovery event=PASS")


def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """告警接口只运行无硬件测试；在线告警由main.py统一调用。"""
    parser = argparse.ArgumentParser(description="跌倒告警接口管理器")
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
    run_self_test(config)


if __name__ == "__main__":
    main()

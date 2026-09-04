# -*- coding: utf-8 -*-
"""豆包视觉模型单图跌倒复核模块。

本文件只负责AI复核，不计算本地P/H/V/S/C：
1. 本地FallScore达到配置阈值后，为当前人员创建一次单图复核请求；
2. 使用后台线程调用火山方舟Responses API，避免网络等待卡住相机；
3. 把模型回答统一转换成FALL、NO_FALL或UNCERTAIN；
4. 请求失败、超时或断网时返回失败结果，主流程自动继续使用本地判断；
5. 保留统一结果结构，后续语音识别、人工确认可接入decision_fusion.py。

本版每次请求只上传一张图片，不再上传多帧序列。上传时机由
AIFallCoordinator控制，图片的裁剪、缩放和JPEG压缩由main.py完成。
"""

import argparse
import base64
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from logging_utils import configure_logging

AI_FALL = "FALL"
AI_NO_FALL = "NO_FALL"
AI_UNCERTAIN = "UNCERTAIN"
logger = logging.getLogger(__name__)


@dataclass
class AIFrameObservation:
    """触发时刻的一人本地测量，以及本次唯一上传的JPEG图片。"""

    person_id: int
    timestamp_s: float
    local_label: str
    local_fall_score: float
    pose_score: float
    height_score: float
    velocity_score: float
    static_score: float
    scene_score: float
    valid_dimensions: Tuple[str, ...]
    degraded_mode: bool
    data_quality: float
    angle_2d_deg: Optional[float]
    angle_3d_deg: Optional[float]
    hip_height_m: Optional[float]
    vertical_velocity_mps: Optional[float]
    static_duration_s: Optional[float]
    scene_relation: str
    scene_confidence: float
    image_jpeg: Optional[bytes] = field(default=None, repr=False)

    @staticmethod
    def _number(value: Optional[float]) -> Optional[float]:
        """有效数字保留三位小数，无效值转成JSON的null。"""
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), 3)

    def to_prompt_dict(self) -> dict:
        """转换为可选的本地辅助信息，不包含图片二进制。"""
        return {
            "person_id": int(self.person_id),
            "local_label": str(self.local_label),
            "local_fall_score": self._number(self.local_fall_score),
            "P_pose": self._number(self.pose_score),
            "H_height": self._number(self.height_score),
            "V_downward_velocity": self._number(self.velocity_score),
            "S_static": self._number(self.static_score),
            "C_scene": self._number(self.scene_score),
            "valid_dimensions": list(self.valid_dimensions),
            "degraded_2d_mode": bool(self.degraded_mode),
            "data_quality": self._number(self.data_quality),
            "angle_2d_deg_0_horizontal": self._number(self.angle_2d_deg),
            "angle_3d_deg_90_horizontal": self._number(self.angle_3d_deg),
            "hip_height_m": self._number(self.hip_height_m),
            "vertical_velocity_mps_negative_down": self._number(
                self.vertical_velocity_mps
            ),
            "static_duration_s": self._number(self.static_duration_s),
            "scene_relation": str(self.scene_relation),
            "scene_confidence": self._number(self.scene_confidence),
        }


@dataclass(frozen=True)
class AIVerificationRequest:
    """一次单图请求；frame.image_jpeg必须只包含当前这一张图片。"""

    event_id: str
    person_id: int
    created_s: float
    frame: AIFrameObservation
    image_url: str = ""
    image_mime_type: str = "image/jpeg"


@dataclass
class AIVerificationResult:
    """统一AI结果；success=False或UNCERTAIN时由融合层退回本地判断。"""

    event_id: str
    person_id: int
    success: bool
    verdict: str
    confidence: float
    risk_level: str
    reason: str
    observations: Tuple[str, ...]
    requested_s: float
    completed_s: float
    latency_s: float
    model: str
    includes_images: bool
    peak_local_score: float
    peak_pose_score: float
    peak_data_quality: float
    raw_answer: str = ""
    error: str = ""

    @property
    def usable(self) -> bool:
        """只有成功得到明确FALL/NO_FALL的结果才覆盖本地结论。"""
        return bool(
            self.success
            and self.verdict in {AI_FALL, AI_NO_FALL}
        )


@dataclass
class AIPersonStatus:
    """main.py画面显示使用的AI运行状态。"""

    state: str
    result: Optional[AIVerificationResult]
    pending: bool
    detail: str


class ArkResponsesClient:
    """按用户验证过的OpenAI SDK写法调用火山方舟Responses API。"""

    def __init__(
        self,
        config: dict,
        transport: Optional[Callable[[dict], object]] = None,
    ):
        self.cfg = config["ai"]
        self.model = str(self.cfg["model"]).strip()
        self.base_url = str(self.cfg["base_url"]).rstrip("/")
        self.api_key_env = str(self.cfg.get("api_key_env", "ARK_API_KEY"))
        self.supports_vision = bool(self.cfg.get("supports_vision", True))
        self.transport = transport
        self._sdk_client = None
        self._sdk_client_key = ""

    @property
    def debug_enabled(self) -> bool:
        """是否输出不包含API Key和Base64正文的安全调试日志。"""
        return bool(self.cfg.get("debug", {}).get("enabled", False))

    @property
    def api_key(self) -> str:
        """优先读取用户填写的ai.api_key，留空时再读取环境变量。"""
        config_key = str(self.cfg.get("api_key", "")).strip()
        if config_key:
            return config_key
        return os.environ.get(self.api_key_env, "").strip()

    @property
    def ready(self) -> bool:
        """离线测试transport或真实API Key存在时才允许提交。"""
        return self.transport is not None or bool(self.api_key)

    def _prompt_text(self, observation: AIFrameObservation) -> str:
        """读取统一提示词；可选附加本地分数，但默认让AI独立看图。"""
        prompt_cfg = self.cfg.get("prompt") or {}
        prompt_text = str(prompt_cfg.get("text", "")).strip()
        if not prompt_text:
            raise ValueError("config.yaml中ai.prompt.text不能为空")
        if bool(prompt_cfg.get("include_local_evidence", False)):
            local_data = json.dumps(
                observation.to_prompt_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            prompt_text += f"\n本地检测辅助数据：{local_data}"
        return prompt_text

    @staticmethod
    def _image_data_url(image_bytes: bytes, mime_type: str) -> str:
        """把本地图片字节编码为Responses API可接收的Data URL。"""
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def build_payload(self, request: AIVerificationRequest) -> dict:
        """构建Responses API参数，每个请求严格只放一张图片。"""
        if not self.supports_vision:
            raise RuntimeError("ai.supports_vision必须为true才能上传图片")
        if request.frame.image_jpeg:
            image_url = self._image_data_url(
                request.frame.image_jpeg,
                request.image_mime_type,
            )
        elif request.image_url:
            image_url = request.image_url
        else:
            raise ValueError("AI单图请求中没有有效图片")

        user_content = [
            {"type": "input_image", "image_url": image_url},
            {
                "type": "input_text",
                "text": self._prompt_text(request.frame),
            },
        ]
        request_cfg = self.cfg["request"]
        return {
            "model": self.model,
            "input": [{"role": "user", "content": user_content}],
            "max_output_tokens": int(request_cfg["max_output_tokens"]),
            "temperature": float(request_cfg["temperature"]),
            "top_p": float(request_cfg.get("top_p", 1.0)),
            "extra_body": {
                "thinking": {
                    "type": (
                        "enabled"
                        if bool(request_cfg.get("thinking_enabled", False))
                        else "disabled"
                    )
                }
            },
            "extra_headers": request_cfg.get("extra_headers", {}) or {},
        }

    def _request_api(self, payload: dict) -> object:
        """执行一次同步SDK调用；在线程序会在线程池中调用这里。"""
        if self.transport is not None:
            return self.transport(payload)

        # 延迟导入使无API环境仍可运行本地检测与--self-test。
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "未安装openai，请执行：python -m pip install -U openai"
            ) from error

        # 复用SDK客户端和HTTP连接，降低连续复核时的握手开销。
        current_key = self.api_key
        if self._sdk_client is None or self._sdk_client_key != current_key:
            self._sdk_client = OpenAI(
                base_url=self.base_url,
                api_key=current_key,
                timeout=float(self.cfg["request"]["timeout_s"]),
                max_retries=0,
            )
            self._sdk_client_key = current_key
        return self._sdk_client.responses.create(**payload)

    @staticmethod
    def _extract_answer_text(response: object) -> str:
        """兼容SDK对象和测试字典，从Responses API结果中提取文本。"""
        output_text = (
            response.get("output_text")
            if isinstance(response, dict)
            else getattr(response, "output_text", None)
        )
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = (
            response.get("output", [])
            if isinstance(response, dict)
            else getattr(response, "output", [])
        ) or []
        for item in output:
            content = (
                item.get("content", [])
                if isinstance(item, dict)
                else getattr(item, "content", [])
            ) or []
            for content_item in content:
                text = (
                    content_item.get("text")
                    if isinstance(content_item, dict)
                    else getattr(content_item, "text", None)
                )
                if isinstance(text, str) and text.strip():
                    return text.strip()
        raise ValueError("Responses API响应中没有可解析文本")

    @staticmethod
    def _strip_code_block(text: str) -> str:
        """移除模型偶尔附加的Markdown代码块标记。"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_answer(self, text: str) -> Tuple[str, float, str, Tuple[str, ...]]:
        """兼容测试文件的true/false，也支持以后扩展为JSON回答。"""
        cleaned = self._strip_code_block(text)
        normalized = cleaned.lower().rstrip("。.!！")
        boolean_confidence = float(
            self.cfg.get("prompt", {}).get("plain_boolean_confidence", 1.0)
        )
        if normalized == "true":
            return AI_FALL, boolean_confidence, "视觉模型判断有人跌倒", ()
        if normalized == "false":
            return AI_NO_FALL, boolean_confidence, "视觉模型未发现跌倒", ()
        if normalized in {"uncertain", "unknown", "不确定"}:
            return AI_UNCERTAIN, 0.0, "视觉证据不足", ()

        try:
            model_data = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError("模型必须只回答true、false或JSON对象") from error
        if not isinstance(model_data, dict):
            raise ValueError("模型JSON回答必须是对象")

        if "is_fall" in model_data:
            value = model_data["is_fall"]
            if isinstance(value, bool):
                verdict = AI_FALL if value else AI_NO_FALL
            else:
                verdict = self._normalize_verdict(value)
        else:
            verdict = self._normalize_verdict(model_data.get("verdict"))
        confidence = float(
            np.clip(
                float(model_data.get("confidence", boolean_confidence)),
                0.0,
                1.0,
            )
        )
        reason = str(model_data.get("reason", "视觉模型已完成判断"))
        observations = model_data.get("observations") or []
        if not isinstance(observations, list):
            observations = [str(observations)]
        return verdict, confidence, reason, tuple(str(item) for item in observations[:5])

    @staticmethod
    def _normalize_verdict(value: object) -> str:
        """把JSON中可能出现的中英文结论统一为三种标准值。"""
        normalized = str(value).strip().lower().replace("-", "_")
        mappings = {
            "true": AI_FALL,
            "fall": AI_FALL,
            "跌倒": AI_FALL,
            "false": AI_NO_FALL,
            "no_fall": AI_NO_FALL,
            "normal": AI_NO_FALL,
            "未跌倒": AI_NO_FALL,
            "正常": AI_NO_FALL,
            "uncertain": AI_UNCERTAIN,
            "unknown": AI_UNCERTAIN,
            "不确定": AI_UNCERTAIN,
        }
        return mappings.get(normalized, AI_UNCERTAIN)

    def verify(self, request: AIVerificationRequest) -> AIVerificationResult:
        """同步完成一次单图复核；任何异常都转换成可降级的失败结果。"""
        started_s = time.monotonic()
        observation = request.frame
        raw_answer = ""
        try:
            if not self.ready:
                raise RuntimeError(
                    f"未找到API Key，请设置{self.api_key_env}或ai.api_key"
                )
            payload = self.build_payload(request)
            if self.debug_enabled:
                image_size_kb = len(observation.image_jpeg or b"") / 1024.0
                logger.info(
                    "AI请求开始：ID=%d event=%s local=%.2f jpeg=%.1fKB model=%s",
                    request.person_id, request.event_id,
                    observation.local_fall_score, image_size_kb, self.model,
                )
            response = self._request_api(payload)
            raw_answer = self._extract_answer_text(response)
            verdict, confidence, reason, observations = self._parse_answer(raw_answer)
            completed_s = time.monotonic()
            if self.debug_enabled:
                logger.info(
                    "AI请求完成：ID=%d verdict=%s confidence=%.2f latency=%.2fs answer=%r",
                    request.person_id, verdict, confidence,
                    completed_s - started_s, raw_answer,
                )
            return AIVerificationResult(
                event_id=request.event_id,
                person_id=request.person_id,
                success=True,
                verdict=verdict,
                confidence=confidence,
                risk_level="high" if verdict == AI_FALL else "low",
                reason=reason,
                observations=observations,
                requested_s=request.created_s,
                completed_s=completed_s,
                latency_s=max(0.0, completed_s - started_s),
                model=self.model,
                includes_images=True,
                peak_local_score=float(observation.local_fall_score),
                peak_pose_score=float(observation.pose_score),
                peak_data_quality=float(observation.data_quality),
                raw_answer=raw_answer,
            )
        except Exception as error:
            completed_s = time.monotonic()
            if self.debug_enabled:
                logger.warning(
                    "AI请求失败，改用本地判断：ID=%d latency=%.2fs error=%s: %s",
                    request.person_id, completed_s - started_s,
                    type(error).__name__, error,
                )
            return AIVerificationResult(
                event_id=request.event_id,
                person_id=request.person_id,
                success=False,
                verdict=AI_UNCERTAIN,
                confidence=0.0,
                risk_level="unknown",
                reason="AI调用失败，系统继续使用本地跌倒判断",
                observations=(),
                requested_s=request.created_s,
                completed_s=completed_s,
                latency_s=max(0.0, completed_s - started_s),
                model=self.model,
                includes_images=bool(observation.image_jpeg or request.image_url),
                peak_local_score=float(observation.local_fall_score),
                peak_pose_score=float(observation.pose_score),
                peak_data_quality=float(observation.data_quality),
                raw_answer=raw_answer,
                error=f"{type(error).__name__}: {error}",
            )


# 旧代码如果导入ArkChatClient仍可继续运行，实际实现已经切换到Responses API。
ArkChatClient = ArkResponsesClient


class AsyncAIVerifier:
    """用后台线程执行网络请求，确保相机主循环不会等待AI。"""

    def __init__(
        self,
        config: dict,
        client: Optional[ArkResponsesClient] = None,
    ):
        self.cfg = config["ai"]
        self.client = client or ArkResponsesClient(config)
        self.max_pending = int(self.cfg["request"]["max_pending_requests"])
        self.executor = ThreadPoolExecutor(
            max_workers=int(self.cfg["request"]["max_workers"]),
            thread_name_prefix="fall-ai",
        )
        self.futures: Dict[Future, AIVerificationRequest] = {}

    def submit(self, request: AIVerificationRequest) -> bool:
        """队列有空间时提交；队列满时跳过，不阻塞主循环。"""
        if len(self.futures) >= self.max_pending:
            return False
        future = self.executor.submit(self.client.verify, request)
        self.futures[future] = request
        return True

    def poll(self) -> List[AIVerificationResult]:
        """无阻塞收集已经完成的结果。"""
        completed: List[AIVerificationResult] = []
        for future, request in list(self.futures.items()):
            if not future.done():
                continue
            del self.futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                now = time.monotonic()
                frame = request.frame
                completed.append(
                    AIVerificationResult(
                        event_id=request.event_id,
                        person_id=request.person_id,
                        success=False,
                        verdict=AI_UNCERTAIN,
                        confidence=0.0,
                        risk_level="unknown",
                        reason="AI后台任务异常，系统继续使用本地判断",
                        observations=(),
                        requested_s=request.created_s,
                        completed_s=now,
                        latency_s=max(0.0, now - request.created_s),
                        model=self.client.model,
                        includes_images=bool(frame.image_jpeg),
                        peak_local_score=float(frame.local_fall_score),
                        peak_pose_score=float(frame.pose_score),
                        peak_data_quality=float(frame.data_quality),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        return completed

    def close(self) -> None:
        """停止接收新请求；正在执行的请求仍受timeout_s限制。"""
        self.executor.shutdown(wait=False, cancel_futures=True)


class AIFallCoordinator:
    """管理0.50触发、单图提交、冷却、断网退避和最新AI结果。"""

    def __init__(
        self,
        config: dict,
        client: Optional[ArkResponsesClient] = None,
    ):
        self.cfg = config["ai"]
        self.enabled = bool(self.cfg["enabled"])
        self.verifier = AsyncAIVerifier(config, client=client)
        self.pending_people: Dict[int, str] = {}
        self.latest_results: Dict[int, AIVerificationResult] = {}
        self.consecutive_risk: Dict[int, int] = defaultdict(int)
        self.last_submit_s: Dict[int, float] = {}
        self.latest_local_score: Dict[int, float] = {}
        self.ignored_event_ids: Set[str] = set()
        self.failure_count = 0
        self.backoff_until_s = 0.0

    @property
    def debug_enabled(self) -> bool:
        """读取AI调试开关。"""
        return bool(self.cfg.get("debug", {}).get("enabled", False))

    @property
    def supports_vision(self) -> bool:
        return bool(self.verifier.client.supports_vision)

    def _risk_triggered(self, observation: AIFrameObservation) -> bool:
        """本地FallScore达到阈值就认为值得上传，不再强制其他维度。"""
        return bool(
            observation.local_fall_score
            >= float(self.cfg["trigger"]["fall_score"])
        )

    def _request_allowed(self, observation: AIFrameObservation) -> bool:
        """检查密钥、视觉能力、连续帧、冷却、队列和网络退避。"""
        person_id = int(observation.person_id)
        now = float(observation.timestamp_s)
        if not self.enabled or not self.verifier.client.ready:
            return False
        if not self.supports_vision:
            return False

        if self._risk_triggered(observation):
            self.consecutive_risk[person_id] += 1
        else:
            self.consecutive_risk[person_id] = 0
            return False

        required_frames = int(self.cfg["trigger"]["consecutive_frames"])
        if self.consecutive_risk[person_id] < required_frames:
            return False
        if person_id in self.pending_people or now < self.backoff_until_s:
            return False

        cooldown_key = (
            "fall_recheck_s"
            if observation.local_label == "FALL"
            else "cooldown_s"
        )
        cooldown_s = float(self.cfg["trigger"][cooldown_key])
        last_submit = self.last_submit_s.get(person_id)
        return bool(last_submit is None or now - last_submit >= cooldown_s)

    def observe(
        self,
        observation: AIFrameObservation,
        image_provider: Optional[Callable[[], Optional[bytes]]] = None,
    ) -> bool:
        """风险达到0.50时获取当前单帧并异步提交；返回是否提交成功。"""
        self.latest_local_score[int(observation.person_id)] = float(
            observation.local_fall_score
        )
        if not self._request_allowed(observation):
            return False

        if not observation.image_jpeg and image_provider is not None:
            observation.image_jpeg = image_provider()
        if not observation.image_jpeg:
            return False

        person_id = int(observation.person_id)
        now = float(observation.timestamp_s)
        event_id = f"person-{person_id}-{int(now * 1000)}"
        verification_request = AIVerificationRequest(
            event_id=event_id,
            person_id=person_id,
            created_s=now,
            frame=observation,
        )
        if not self.verifier.submit(verification_request):
            if self.debug_enabled:
                logger.warning("AI请求队列已满，本次跳过：ID=%d", person_id)
            return False

        self.pending_people[person_id] = event_id
        self.last_submit_s[person_id] = now
        self.consecutive_risk[person_id] = 0
        if self.debug_enabled:
            logger.info(
                "AI单图请求已提交：ID=%d event=%s FallScore=%.2f",
                person_id, event_id, observation.local_fall_score,
            )
        return True

    def poll(self, now_s: Optional[float] = None) -> List[AIVerificationResult]:
        """收集AI结果；失败后指数退避，期间直接跳过新AI请求。"""
        now = time.monotonic() if now_s is None else float(now_s)
        accepted_results: List[AIVerificationResult] = []
        for result in self.verifier.poll():
            if result.event_id in self.ignored_event_ids:
                self.ignored_event_ids.discard(result.event_id)
                continue
            if self.pending_people.get(result.person_id) == result.event_id:
                self.pending_people.pop(result.person_id, None)
            self.latest_results[result.person_id] = result
            accepted_results.append(result)

            if result.success:
                self.failure_count = 0
                self.backoff_until_s = 0.0
            else:
                self.failure_count += 1
                network_cfg = self.cfg["network"]
                delay = min(
                    float(network_cfg["backoff_initial_s"])
                    * (2 ** max(0, self.failure_count - 1)),
                    float(network_cfg["backoff_max_s"]),
                )
                self.backoff_until_s = now + delay
                if self.debug_enabled:
                    logger.warning(
                        "AI进入网络退避：连续失败=%d，暂停=%.1fs",
                        self.failure_count, delay,
                    )
        return accepted_results

    def latest_result(
        self,
        person_id: int,
        now_s: float,
    ) -> Optional[AIVerificationResult]:
        """只返回有效期内、成功且结论明确的AI结果。"""
        result = self.latest_results.get(int(person_id))
        if result is None or not result.usable:
            return None
        ttl_s = float(self.cfg["trigger"]["result_ttl_s"])
        if float(now_s) - result.completed_s > ttl_s:
            return None
        return result

    def status(self, person_id: int, now_s: float) -> AIPersonStatus:
        """生成画面状态，不显示API Key或完整请求内容。"""
        person_id = int(person_id)
        now = float(now_s)
        if not self.enabled:
            return AIPersonStatus("DISABLED", None, False, "配置已关闭")
        if not self.supports_vision:
            return AIPersonStatus(
                "NO_VISION",
                None,
                False,
                "ai.supports_vision必须为true",
            )
        if not self.verifier.client.ready:
            return AIPersonStatus(
                "NO_KEY",
                None,
                False,
                f"请设置{self.verifier.client.api_key_env}或ai.api_key",
            )
        if person_id in self.pending_people:
            return AIPersonStatus("PENDING", None, True, "单图后台复核中")
        result = self.latest_results.get(person_id)
        if now < self.backoff_until_s:
            remaining = max(0.0, self.backoff_until_s - now)
            return AIPersonStatus(
                "BACKOFF",
                result,
                False,
                f"网络失败，{remaining:.1f}秒内跳过AI",
            )
        local_score = self.latest_local_score.get(person_id)
        trigger_score = float(self.cfg["trigger"]["fall_score"])
        if local_score is not None and local_score < trigger_score:
            return AIPersonStatus(
                "LOCAL_LOW",
                result,
                False,
                f"本地分数{local_score:.2f}低于触发值{trigger_score:.2f}",
            )
        last_submit = self.last_submit_s.get(person_id)
        if last_submit is not None:
            cooldown_s = float(self.cfg["trigger"]["cooldown_s"])
            remaining = cooldown_s - (now - last_submit)
            if remaining > 0.0 and result is None:
                return AIPersonStatus(
                    "COOLDOWN",
                    None,
                    False,
                    f"等待{remaining:.1f}秒后允许再次复核",
                )
        if result is not None and not result.success:
            return AIPersonStatus("ERROR", result, False, result.error)
        if (
            result is not None
            and result.success
            and result.verdict == AI_UNCERTAIN
            and now - result.completed_s
            <= float(self.cfg["trigger"]["result_ttl_s"])
        ):
            return AIPersonStatus(
                AI_UNCERTAIN,
                result,
                False,
                result.reason,
            )
        usable = self.latest_result(person_id, now)
        if usable is not None:
            return AIPersonStatus(usable.verdict, usable, False, usable.reason)
        return AIPersonStatus("IDLE", None, False, "等待本地分数达到触发值")

    def reset_person(self, person_id: int) -> None:
        """人员离场时清除AI状态，迟到的旧结果不会污染复用ID。"""
        person_id = int(person_id)
        pending_event = self.pending_people.pop(person_id, None)
        if pending_event is not None:
            self.ignored_event_ids.add(pending_event)
        self.latest_results.pop(person_id, None)
        self.consecutive_risk.pop(person_id, None)
        self.last_submit_s.pop(person_id, None)
        self.latest_local_score.pop(person_id, None)

    def close(self) -> None:
        """关闭后台线程池。"""
        self.verifier.close()


def _test_observation(image_jpeg: bytes = b"fake-jpeg") -> AIFrameObservation:
    """构造离线测试使用的单帧高风险观测。"""
    return AIFrameObservation(
        person_id=1,
        timestamp_s=10.0,
        local_label="NO_FALL",
        local_fall_score=0.50,
        pose_score=0.85,
        height_score=0.70,
        velocity_score=0.60,
        static_score=0.40,
        scene_score=1.00,
        valid_dimensions=("P", "H", "V", "S", "C"),
        degraded_mode=False,
        data_quality=0.90,
        angle_2d_deg=10.0,
        angle_3d_deg=78.0,
        hip_height_m=0.20,
        vertical_velocity_mps=-0.80,
        static_duration_s=1.2,
        scene_relation="lying_on_floor",
        scene_confidence=0.85,
        image_jpeg=image_jpeg,
    )


def run_self_test(config: dict) -> None:
    """验证Responses格式、0.50触发、单图、异步结果和断网降级。"""
    test_config = json.loads(json.dumps(config))
    test_config["ai"]["enabled"] = True
    test_config["ai"]["supports_vision"] = True
    test_config["ai"]["trigger"]["fall_score"] = 0.50
    test_config["ai"]["trigger"]["consecutive_frames"] = 1

    def fake_transport(payload: dict) -> dict:
        content = payload["input"][0]["content"]
        image_items = [item for item in content if item["type"] == "input_image"]
        assert len(image_items) == 1
        assert image_items[0]["image_url"].startswith("data:image/jpeg;base64,")
        return {"output_text": "true"}

    client = ArkResponsesClient(test_config, transport=fake_transport)
    coordinator = AIFallCoordinator(test_config, client=client)
    observation = _test_observation(image_jpeg=b"")
    image_provider_calls = []

    def image_provider() -> bytes:
        image_provider_calls.append(1)
        return b"one-jpeg-only"

    assert coordinator.observe(observation, image_provider=image_provider)
    assert len(image_provider_calls) == 1

    results: List[AIVerificationResult] = []
    for _ in range(100):
        results = coordinator.poll(now_s=10.1)
        if results:
            break
        time.sleep(0.01)
    assert len(results) == 1
    assert results[0].usable and results[0].verdict == AI_FALL
    assert results[0].includes_images

    # 低于0.50不能编码图片，也不能提交API。
    low_risk = _test_observation(image_jpeg=b"")
    low_risk.person_id = 2
    low_risk.local_fall_score = 0.49
    assert not coordinator.observe(low_risk, image_provider=image_provider)
    assert len(image_provider_calls) == 1
    coordinator.close()

    # 模拟断网：失败结果必须被收集，不能向主循环抛异常。
    offline_client = ArkResponsesClient(
        test_config,
        transport=lambda payload: (_ for _ in ()).throw(
            RuntimeError("NETWORK_ERROR")
        ),
    )
    offline_result = offline_client.verify(
        AIVerificationRequest(
            event_id="offline-test",
            person_id=1,
            created_s=10.0,
            frame=_test_observation(),
        )
    )
    assert not offline_result.success
    assert offline_result.verdict == AI_UNCERTAIN
    print("ai_verifier self-test: PASS")
    print("  Responses API, 0.50 trigger, one image, async and offline fallback=PASS")


def _load_image_input(image_input: str, config: dict,) -> Tuple[bytes, str, str]:
    """读取测试图片；本地图片会缩小并压缩成JPEG，URL保持不变。"""

    # 网络图片和已经编码好的Data URL不需要本地处理。
    if image_input.startswith(("http://", "https://", "data:image/")):
        return b"", image_input, "image/jpeg"

    image_path = Path(image_input).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"测试图片不存在：{image_path}")

    # 放在函数内导入，不影响不使用真实图片的离线测试。
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "OpenCV无法读取图片，请使用jpg、jpeg、png或webp格式"
        )

    image_config = config["ai"]["image"]
    max_long_side = int(image_config["max_long_side_px"])
    jpeg_quality = int(image_config["jpeg_quality"])

    if max_long_side <= 0:
        raise ValueError("ai.image.max_long_side_px必须大于0")

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("ai.image.jpeg_quality必须在1到100之间")

    # 图片最长边超过限制时等比例缩小，小图片不放大。
    image_height, image_width = image.shape[:2]
    current_long_side = max(image_height, image_width)

    if current_long_side > max_long_side:
        scale = max_long_side / float(current_long_side)
        output_width = max(1, int(round(image_width * scale)))
        output_height = max(1, int(round(image_height * scale)))

        image = cv2.resize(
            image,
            (output_width, output_height),
            interpolation=cv2.INTER_AREA,
        )

    # 无论原图是PNG还是JPG，都统一压缩为较小的JPEG。
    success, encoded_image = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not success:
        raise RuntimeError("测试图片JPEG编码失败")

    image_bytes = encoded_image.tobytes()

    print(
        f"图片处理完成：{image.shape[1]}×{image.shape[0]}，"
        f"JPEG大小约{len(image_bytes) / 1024:.1f}KB"
    )

    return image_bytes, "", "image/jpeg" 


def run_api_test(config: dict, image_input: str) -> None:
    """上传用户指定的一张真实图片，验证模型能否返回true或false。"""
    client = ArkResponsesClient(config)
    if not client.ready:
        raise RuntimeError(
            f"请先设置{client.api_key_env}或config.yaml中的ai.api_key"
        )
    image_bytes, image_url, mime_type = _load_image_input(
    image_input,
    config,
    )
    observation = _test_observation(image_jpeg=image_bytes)
    now = time.monotonic()
    observation.person_id = 999
    observation.timestamp_s = now
    request = AIVerificationRequest(
        event_id="manual-single-image-test",
        person_id=999,
        created_s=now,
        frame=observation,
        image_url=image_url,
        image_mime_type=mime_type,
    )
    result = client.verify(request)
    print(
        f"API test: success={result.success} "
        f"verdict={result.verdict} confidence={result.confidence:.2f}"
    )
    print(f"answer={result.raw_answer or '<empty>'}")
    print(f"reason={result.reason}")
    if not result.success:
        raise RuntimeError(result.error)


def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """支持离线自测、真实单图API测试和相机AI模式。"""
    parser = argparse.ArgumentParser(description="豆包视觉模型单图跌倒复核")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="统一配置文件路径",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="运行不访问真实API的合成测试",
    )
    parser.add_argument(
        "--api-test",
        metavar="IMAGE",
        help="上传一张本地图片、图片URL或Data URL进行真实API测试",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config, Path(__file__).resolve().parent)
    if args.self_test:
        run_self_test(config)
    elif args.api_test:
        run_api_test(config, args.api_test)
    else:
        from main import run_live

        run_live(config, args.config, stage="ai")


if __name__ == "__main__":
    main()

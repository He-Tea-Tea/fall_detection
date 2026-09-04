# -*- coding: utf-8 -*-
"""豆包AI跌倒复核模块。

本文件负责四件事：
1. 按Track ID保存最近几秒的本地检测结果；
2. 本地风险达到配置阈值时，生成一次AI复核事件；
3. 在后台线程调用火山方舟兼容Chat API，不阻塞相机主循环；
4. 把模型返回内容解析成统一的FALL、NO_FALL或UNCERTAIN结果。

当前配置模型doubao-1-5-pro-32k-250115是文本模型，因此本版主要发送
P/H/V/S/C、人体角度、髋高、下降速度、静止时间和场景关系。代码已经
预留多帧JPEG输入；以后改用视觉模型时，只需修改config.yaml中的模型名，
并把supports_vision改为true，不需要重写main.py调用流程。

安全原则：
- API密钥只从环境变量读取，绝不写进代码或config.yaml；
- 网络错误、超时、模型错误都不会中断本地跌倒检测；
- AI只返回复核意见，最终是否报警由decision_fusion.py统一决定；
- 同一人员使用事件触发和冷却时间，避免每帧重复请求产生费用。
"""

import argparse
import base64
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import yaml

AI_FALL = "FALL"
AI_NO_FALL = "NO_FALL"
AI_UNCERTAIN = "UNCERTAIN"


@dataclass
class AIFrameObservation:
    """一个人在某个时间点的本地测量，可选保存对应人体区域JPEG。"""

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
        """把有效数字保留三位小数，无效数字转换成JSON的null。"""
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), 3)

    def to_prompt_dict(self, first_timestamp_s: float) -> dict:
        """转换成发送给AI的精简字典，不包含二进制图片。"""
        return {
            "time_from_first_s": round(
                float(self.timestamp_s) - float(first_timestamp_s),
                3,
            ),
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


@dataclass
class AIVerificationRequest:
    """一次不可变的AI复核请求；frames按时间从早到晚排列。"""

    event_id: str
    person_id: int
    created_s: float
    frames: Tuple[AIFrameObservation, ...]


@dataclass
class AIVerificationResult:
    """统一AI复核结果；success=False时不能参与最终判断。"""

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
    error: str = ""

    @property
    def usable(self) -> bool:
        """只有成功解析出的三种标准结论才允许进入融合层。"""
        return bool(
            self.success
            and self.verdict in {AI_FALL, AI_NO_FALL, AI_UNCERTAIN}
        )


@dataclass
class AIPersonStatus:
    """main.py用于画面显示的AI运行状态。"""

    state: str
    result: Optional[AIVerificationResult]
    pending: bool
    detail: str


class AIEventBuffer:
    """按Track ID保存短时间序列，并从中选择有代表性的关键帧。"""

    def __init__(self, config: dict):
        self.cfg = config["ai"]["buffer"]
        self.duration_s = float(self.cfg["duration_s"])
        self.sample_interval_s = float(self.cfg["sample_interval_s"])
        self.keyframe_count = int(self.cfg["keyframe_count"])
        self.max_frames = int(self.cfg["max_frames_per_person"])
        if self.duration_s <= 0.0 or self.sample_interval_s <= 0.0:
            raise ValueError("ai.buffer的时长和采样间隔必须大于0")
        if self.keyframe_count <= 0 or self.max_frames < self.keyframe_count:
            raise ValueError("ai.buffer最大帧数不能小于关键帧数量")
        self.people: Dict[int, Deque[AIFrameObservation]] = defaultdict(
            lambda: deque(maxlen=self.max_frames)
        )
        self.last_sample_s: Dict[int, float] = {}

    def should_capture_image(self, person_id: int, timestamp_s: float) -> bool:
        """告诉main.py本帧是否值得编码JPEG，减少不必要的CPU消耗。"""
        last_sample = self.last_sample_s.get(int(person_id))
        return bool(
            last_sample is None
            or float(timestamp_s) - last_sample >= self.sample_interval_s
        )

    def add(self, observation: AIFrameObservation) -> bool:
        """按采样间隔保存观测；返回True表示本帧已进入缓冲区。"""
        person_id = int(observation.person_id)
        now = float(observation.timestamp_s)
        last_sample = self.last_sample_s.get(person_id)
        if (
            last_sample is not None
            and now - last_sample < self.sample_interval_s
        ):
            return False

        history = self.people[person_id]
        history.append(observation)
        self.last_sample_s[person_id] = now

        # maxlen限制条数，duration_s进一步限制真实时间跨度。
        while history and now - history[0].timestamp_s > self.duration_s:
            history.popleft()
        return True

    def _keyframe_indices(self, frames: Sequence[AIFrameObservation]) -> List[int]:
        """优先保留最早、风险最高和最新帧，再均匀补足其他帧。"""
        frame_count = len(frames)
        if frame_count <= self.keyframe_count:
            return list(range(frame_count))

        selected = {0, frame_count - 1}
        peak_index = max(
            range(frame_count),
            key=lambda index: frames[index].local_fall_score,
        )
        selected.add(peak_index)

        # 均匀候选保证AI能看见事件发展过程，而不是只看最高分一帧。
        evenly_spaced = np.linspace(
            0,
            frame_count - 1,
            num=self.keyframe_count,
        ).round().astype(int)
        for index in evenly_spaced:
            selected.add(int(index))
            if len(selected) >= self.keyframe_count:
                break

        # 如果均匀点与峰值重复，从中间向两侧继续补足。
        if len(selected) < self.keyframe_count:
            for index in range(1, frame_count - 1):
                selected.add(index)
                if len(selected) >= self.keyframe_count:
                    break
        return sorted(selected)[: self.keyframe_count]

    def build_request(
        self,
        person_id: int,
        created_s: float,
    ) -> Optional[AIVerificationRequest]:
        """从指定人员缓冲区构建一次AI请求，没有观测时返回None。"""
        frames = list(self.people.get(int(person_id), ()))
        if not frames:
            return None
        selected = tuple(frames[index] for index in self._keyframe_indices(frames))
        event_id = f"person-{int(person_id)}-{int(float(created_s) * 1000)}"
        return AIVerificationRequest(
            event_id=event_id,
            person_id=int(person_id),
            created_s=float(created_s),
            frames=selected,
        )

    def reset_person(self, person_id: int) -> None:
        """清除离场人员的图片和结构化历史。"""
        self.people.pop(int(person_id), None)
        self.last_sample_s.pop(int(person_id), None)


class ArkChatClient:
    """使用标准库调用火山方舟兼容Chat Completions接口。"""

    def __init__(
        self,
        config: dict,
        transport: Optional[Callable[[dict], dict]] = None,
    ):
        self.cfg = config["ai"]
        self.model = str(self.cfg["model"])
        self.base_url = str(self.cfg["base_url"]).rstrip("/")
        self.api_key_env = str(self.cfg["api_key_env"])
        self.supports_vision = bool(self.cfg["supports_vision"])
        self.transport = transport

    # @property
    # def api_key(self) -> str:
    #     """优先读取config.yaml中的密钥；没有填写时再读取环境变量。"""
    #     config_key = str(self.cfg.get("api_key", "")).strip()
    #     if config_key:
    #         return config_key
        # return os.environ.get(self.api_key_env, "").strip()

    @property
    def api_key(self) -> str:
        # 直接取，不提前转 str
        config_key = self.cfg.get("api_key")
        # 判断：既不是 None，也不是空字符串，并且去除空格后还有内容
        if config_key is not None and str(config_key).strip():
            return str(config_key).strip()
        # 回退到环境变量
        return os.environ.get(self.api_key_env, "").strip()

    @property
    def ready(self) -> bool:
        """测试transport或真实API密钥任一存在即可发起请求。"""
        return self.transport is not None or bool(self.api_key)

    def _prompt(self, request: AIVerificationRequest) -> str:
        """把本地时间序列整理成明确、可解析的跌倒复核任务。"""
        first_timestamp = request.frames[0].timestamp_s
        sequence = [
            frame.to_prompt_dict(first_timestamp)
            for frame in request.frames
        ]
        payload = {
            "task": "fall_verification",
            "person_id": request.person_id,
            "definitions": {
                "P": "姿态风险，越接近水平越高",
                "H": "髋部下降或绝对低高度风险",
                "V": "快速向下运动风险",
                "S": "异常姿态后持续静止风险",
                "C": "地面比床、沙发等场景更危险",
                "score_range": "所有风险分为0到1",
            },
            "chronological_observations": sequence,
        }
        return (
            "请根据以下按时间排序的本地跌倒检测数据进行二次复核。"
            "不要把单帧水平姿态直接认定为跌倒，要区分跌倒、坐下、弯腰、"
            "床或沙发上正常躺卧以及证据不足。无效维度已经从valid_dimensions"
            "中移除，不能把缺失数据当成正常证据。只输出一个JSON对象，格式为："
            '{"verdict":"fall|no_fall|uncertain","confidence":0到1,'
            '"risk_level":"high|medium|low","reason":"简短中文原因",'
            '"observations":["关键证据1","关键证据2"]}。\n'
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    def _messages(self, request: AIVerificationRequest) -> Tuple[List[dict], bool]:
        """构建兼容OpenAI格式的消息；视觉模式会追加Base64 JPEG。"""
        system_text = (
            "你是服务机器人跌倒检测的保守复核器。你只能根据输入证据判断，"
            "证据冲突或不足必须输出uncertain。不要输出JSON以外的文字。"
        )
        prompt = self._prompt(request)
        image_frames = [
            frame
            for frame in request.frames
            if frame.image_jpeg
        ]
        includes_images = bool(self.supports_vision and image_frames)

        if includes_images:
            content: List[dict] = [{"type": "text", "text": prompt}]
            for frame in image_frames:
                encoded = base64.b64encode(frame.image_jpeg).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                        },
                    }
                )
            user_content = content
        else:
            user_content = prompt

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]
        return messages, includes_images

    def _http_post(self, payload: dict) -> dict:
        """执行一次HTTPS POST；任何异常都交给上层转换成失败结果。"""
        request_url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = urllib_request.Request(
            request_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout_s = float(self.cfg["request"]["timeout_s"])
        try:
            with urllib_request.urlopen(http_request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {error.code}: {response_text}") from error
        except urllib_error.URLError as error:
            raise RuntimeError(f"NETWORK_ERROR: {error.reason}") from error

    @staticmethod
    def _extract_message_text(response: dict) -> str:
        """兼容字符串content和部分服务返回的分段content。"""
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("API响应中没有choices")
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            ]
            return "".join(parts).strip()
        raise ValueError("API响应中没有可解析的message.content")

    @staticmethod
    def _parse_model_json(text: str) -> dict:
        """解析严格JSON，同时兼容模型偶尔返回的Markdown代码块。"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match is None:
                raise ValueError("模型没有返回JSON对象")
            return json.loads(match.group(0))

    @staticmethod
    def _normalize_verdict(value: object) -> str:
        """把模型可能返回的中英文结论统一为三个标准值。"""
        normalized = str(value).strip().lower().replace("-", "_")
        mappings = {
            "fall": AI_FALL,
            "跌倒": AI_FALL,
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
        """同步执行一次AI复核；正常在线流程由后台线程调用本方法。"""
        started_s = time.monotonic()
        messages, includes_images = self._messages(request)
        peak_local_score = max(frame.local_fall_score for frame in request.frames)
        peak_pose_score = max(frame.pose_score for frame in request.frames)
        peak_data_quality = max(frame.data_quality for frame in request.frames)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.cfg["request"]["temperature"]),
            "max_tokens": int(self.cfg["request"]["max_tokens"]),
        }
        if bool(self.cfg["request"].get("force_json_response", False)):
            payload["response_format"] = {"type": "json_object"}

        try:
            if not self.ready:
                raise RuntimeError(f"环境变量{self.api_key_env}尚未设置")
            response = (
                self.transport(payload)
                if self.transport is not None
                else self._http_post(payload)
            )
            model_data = self._parse_model_json(
                self._extract_message_text(response)
            )
            verdict = self._normalize_verdict(model_data.get("verdict"))
            confidence = float(
                np.clip(float(model_data.get("confidence", 0.0)), 0.0, 1.0)
            )
            observations = model_data.get("observations") or []
            if not isinstance(observations, list):
                observations = [str(observations)]
            completed_s = time.monotonic()
            return AIVerificationResult(
                event_id=request.event_id,
                person_id=request.person_id,
                success=True,
                verdict=verdict,
                confidence=confidence,
                risk_level=str(model_data.get("risk_level", "unknown")),
                reason=str(model_data.get("reason", "")),
                observations=tuple(str(item) for item in observations[:5]),
                requested_s=request.created_s,
                completed_s=completed_s,
                latency_s=max(0.0, completed_s - started_s),
                model=self.model,
                includes_images=includes_images,
                peak_local_score=float(peak_local_score),
                peak_pose_score=float(peak_pose_score),
                peak_data_quality=float(peak_data_quality),
            )
        except Exception as error:
            completed_s = time.monotonic()
            return AIVerificationResult(
                event_id=request.event_id,
                person_id=request.person_id,
                success=False,
                verdict=AI_UNCERTAIN,
                confidence=0.0,
                risk_level="unknown",
                reason="AI复核失败，本地检测继续运行",
                observations=(),
                requested_s=request.created_s,
                completed_s=completed_s,
                latency_s=max(0.0, completed_s - started_s),
                model=self.model,
                includes_images=includes_images,
                peak_local_score=float(peak_local_score),
                peak_pose_score=float(peak_pose_score),
                peak_data_quality=float(peak_data_quality),
                error=f"{type(error).__name__}: {error}",
            )


class AsyncAIVerifier:
    """用固定大小线程池执行AI请求，防止网络等待卡住相机取帧。"""

    def __init__(
        self,
        config: dict,
        client: Optional[ArkChatClient] = None,
    ):
        self.cfg = config["ai"]
        self.client = client or ArkChatClient(config)
        self.max_pending = int(self.cfg["request"]["max_pending_requests"])
        self.executor = ThreadPoolExecutor(
            max_workers=int(self.cfg["request"]["max_workers"]),
            thread_name_prefix="fall-ai",
        )
        self.futures: Dict[Future, AIVerificationRequest] = {}

    def submit(self, request: AIVerificationRequest) -> bool:
        """队列有空间时提交请求；队列满时返回False而不是阻塞。"""
        if len(self.futures) >= self.max_pending:
            return False
        future = self.executor.submit(self.client.verify, request)
        self.futures[future] = request
        return True

    def poll(self) -> List[AIVerificationResult]:
        """无阻塞收集已完成请求；未完成请求留到下一帧继续检查。"""
        completed: List[AIVerificationResult] = []
        for future, request in list(self.futures.items()):
            if not future.done():
                continue
            del self.futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                now = time.monotonic()
                completed.append(
                    AIVerificationResult(
                        event_id=request.event_id,
                        person_id=request.person_id,
                        success=False,
                        verdict=AI_UNCERTAIN,
                        confidence=0.0,
                        risk_level="unknown",
                        reason="AI后台任务异常，本地检测继续运行",
                        observations=(),
                        requested_s=request.created_s,
                        completed_s=now,
                        latency_s=max(0.0, now - request.created_s),
                        model=self.client.model,
                        includes_images=False,
                        peak_local_score=max(
                            frame.local_fall_score
                            for frame in request.frames
                        ),
                        peak_pose_score=max(
                            frame.pose_score
                            for frame in request.frames
                        ),
                        peak_data_quality=max(
                            frame.data_quality
                            for frame in request.frames
                        ),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        return completed

    def close(self) -> None:
        """停止接收新任务；正在执行的网络请求受timeout_s限制。"""
        self.executor.shutdown(wait=False, cancel_futures=True)


class AIFallCoordinator:
    """管理事件触发、冷却、网络退避、最新结果和每人请求状态。"""

    def __init__(
        self,
        config: dict,
        client: Optional[ArkChatClient] = None,
    ):
        self.cfg = config["ai"]
        self.enabled = bool(self.cfg["enabled"])
        self.buffer = AIEventBuffer(config)
        self.verifier = AsyncAIVerifier(config, client=client)
        self.pending_people: Dict[int, str] = {}
        self.latest_results: Dict[int, AIVerificationResult] = {}
        self.consecutive_risk: Dict[int, int] = defaultdict(int)
        self.last_submit_s: Dict[int, float] = {}
        self.ignored_event_ids: Set[str] = set()
        self.failure_count = 0
        self.backoff_until_s = 0.0

    @property
    def supports_vision(self) -> bool:
        return bool(self.verifier.client.supports_vision)

    def should_capture_image(self, person_id: int, timestamp_s: float) -> bool:
        """文本模型返回False；视觉模式按缓冲采样间隔返回True。"""
        return bool(
            self.enabled
            and self.supports_vision
            and self.buffer.should_capture_image(person_id, timestamp_s)
        )

    def _risk_triggered(self, observation: AIFrameObservation) -> bool:
        """本地FALL立即触发；疑似风险需要同时满足总分、P和质量门槛。"""
        trigger_cfg = self.cfg["trigger"]
        if observation.local_label == "FALL":
            return True
        return bool(
            observation.local_fall_score >= float(trigger_cfg["fall_score"])
            and observation.pose_score >= float(trigger_cfg["min_pose_score"])
            and observation.data_quality >= float(trigger_cfg["min_data_quality"])
        )

    def observe(self, observation: AIFrameObservation) -> bool:
        """保存本帧并按条件异步提交AI；返回True表示成功创建请求。"""
        self.buffer.add(observation)
        person_id = int(observation.person_id)
        now = float(observation.timestamp_s)

        if not self.enabled or not self.verifier.client.ready:
            return False

        if self._risk_triggered(observation):
            self.consecutive_risk[person_id] += 1
        else:
            self.consecutive_risk[person_id] = 0
            return False

        required_frames = int(self.cfg["trigger"]["consecutive_frames"])
        if observation.local_label != "FALL" and self.consecutive_risk[person_id] < required_frames:
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
        if last_submit is not None and now - last_submit < cooldown_s:
            return False

        verification_request = self.buffer.build_request(person_id, now)
        if verification_request is None or not self.verifier.submit(verification_request):
            return False

        self.pending_people[person_id] = verification_request.event_id
        self.last_submit_s[person_id] = now
        self.consecutive_risk[person_id] = 0
        return True

    def poll(self, now_s: Optional[float] = None) -> List[AIVerificationResult]:
        """收集后台结果，并根据成功或失败更新网络退避状态。"""
        now = time.monotonic() if now_s is None else float(now_s)
        results = self.verifier.poll()
        accepted_results: List[AIVerificationResult] = []
        for result in results:
            # 人员离场后旧网络任务可能才返回，不能让旧结果污染复用的Track ID。
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
        return accepted_results

    def latest_result(
        self,
        person_id: int,
        now_s: float,
    ) -> Optional[AIVerificationResult]:
        """只返回仍在有效期内且成功的结果，过期结果不参与融合。"""
        result = self.latest_results.get(int(person_id))
        if result is None or not result.usable:
            return None
        result_ttl_s = float(self.cfg["trigger"]["result_ttl_s"])
        if float(now_s) - result.completed_s > result_ttl_s:
            return None
        return result

    def status(self, person_id: int, now_s: float) -> AIPersonStatus:
        """生成面向画面的状态，不暴露API密钥或完整响应。"""
        person_id = int(person_id)
        now = float(now_s)
        if not self.enabled:
            return AIPersonStatus("DISABLED", None, False, "配置已关闭")
        if not self.verifier.client.ready:
            return AIPersonStatus(
                "NO_KEY",
                None,
                False,
                f"请设置{self.verifier.client.api_key_env}",
            )
        if person_id in self.pending_people:
            return AIPersonStatus("PENDING", None, True, "后台复核中")
        result = self.latest_results.get(person_id)
        if result is not None and not result.success:
            return AIPersonStatus("ERROR", result, False, result.error)
        usable = self.latest_result(person_id, now)
        if usable is not None:
            return AIPersonStatus(usable.verdict, usable, False, usable.reason)
        if now < self.backoff_until_s:
            remaining = max(0.0, self.backoff_until_s - now)
            return AIPersonStatus(
                "BACKOFF",
                None,
                False,
                f"网络退避{remaining:.1f}s",
            )
        return AIPersonStatus("IDLE", None, False, "等待本地疑似事件")

    def reset_person(self, person_id: int) -> None:
        """人员离场时清除缓冲和AI状态；已提交网络任务不能强制中断。"""
        person_id = int(person_id)
        self.buffer.reset_person(person_id)
        pending_event = self.pending_people.pop(person_id, None)
        if pending_event is not None:
            self.ignored_event_ids.add(pending_event)
        self.latest_results.pop(person_id, None)
        self.consecutive_risk.pop(person_id, None)
        self.last_submit_s.pop(person_id, None)

    def close(self) -> None:
        """释放后台线程池。"""
        self.verifier.close()


def run_self_test(config: dict) -> None:
    """使用假API验证缓冲、异步请求、JSON解析和文本模式。"""
    def fake_transport(payload: dict) -> dict:
        assert payload["model"] == config["ai"]["model"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "verdict": "fall",
                                "confidence": 0.91,
                                "risk_level": "high",
                                "reason": "姿态、高度和静止证据连续异常",
                                "observations": ["P较高", "H较高"],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    test_config = json.loads(json.dumps(config))
    test_config["ai"]["enabled"] = True
    test_config["ai"]["supports_vision"] = False
    test_config["ai"]["trigger"]["consecutive_frames"] = 1
    client = ArkChatClient(test_config, transport=fake_transport)
    coordinator = AIFallCoordinator(test_config, client=client)
    observation = AIFrameObservation(
        person_id=1,
        timestamp_s=10.0,
        local_label="NO_FALL",
        local_fall_score=0.72,
        pose_score=0.90,
        height_score=0.80,
        velocity_score=0.60,
        static_score=0.70,
        scene_score=1.00,
        valid_dimensions=("P", "H", "V", "S", "C"),
        degraded_mode=False,
        data_quality=0.95,
        angle_2d_deg=8.0,
        angle_3d_deg=80.0,
        hip_height_m=0.18,
        vertical_velocity_mps=-0.90,
        static_duration_s=2.0,
        scene_relation="lying_on_floor",
        scene_confidence=0.90,
    )
    assert coordinator.observe(observation)

    results: List[AIVerificationResult] = []
    for _ in range(100):
        results = coordinator.poll(now_s=10.1)
        if results:
            break
        time.sleep(0.01)
    assert len(results) == 1
    assert results[0].success and results[0].verdict == AI_FALL
    assert results[0].confidence > 0.90
    assert not results[0].includes_images
    assert coordinator.latest_result(1, results[0].completed_s) is not None
    coordinator.close()

    # 验证未来视觉模型会使用同一接口附加Base64 JPEG。
    vision_config = json.loads(json.dumps(test_config))
    vision_config["ai"]["supports_vision"] = True
    vision_client = ArkChatClient(vision_config, transport=fake_transport)
    observation.image_jpeg = b"fake-jpeg-bytes"
    vision_request = AIVerificationRequest(
        event_id="vision-interface-test",
        person_id=1,
        created_s=10.0,
        frames=(observation,),
    )
    messages, includes_images = vision_client._messages(vision_request)
    assert includes_images
    assert messages[1]["content"][1]["type"] == "image_url"

    # 模型没有返回JSON时必须形成失败结果，不能抛出异常中断主循环。
    bad_client = ArkChatClient(
        test_config,
        transport=lambda payload: {
            "choices": [{"message": {"content": "无法判断"}}]
        },
    )
    bad_result = bad_client.verify(vision_request)
    assert not bad_result.success and bad_result.verdict == AI_UNCERTAIN
    print("ai_verifier self-test: PASS")
    print(
        "  event buffer, async request, JSON parsing, "
        "vision interface, failure fallback=PASS"
    )


def run_api_test(config: dict) -> None:
    """使用环境变量中的真实密钥发送一次最小请求，便于先验证API。"""
    client = ArkChatClient(config)
    if not client.ready:
        raise RuntimeError(
            f"请先设置环境变量{client.api_key_env}，再运行--api-test"
        )

    now = time.monotonic()
    observation = AIFrameObservation(
        person_id=999,
        timestamp_s=now,
        local_label="NO_FALL",
        local_fall_score=0.68,
        pose_score=0.85,
        height_score=0.75,
        velocity_score=0.60,
        static_score=0.50,
        scene_score=1.00,
        valid_dimensions=("P", "H", "V", "S", "C"),
        degraded_mode=False,
        data_quality=0.90,
        angle_2d_deg=12.0,
        angle_3d_deg=76.0,
        hip_height_m=0.20,
        vertical_velocity_mps=-0.75,
        static_duration_s=1.5,
        scene_relation="lying_on_floor",
        scene_confidence=0.85,
    )
    verification_request = AIVerificationRequest(
        event_id="manual-api-test",
        person_id=999,
        created_s=now,
        frames=(observation,),
    )
    result = client.verify(verification_request)
    print(
        f"API test: success={result.success} "
        f"verdict={result.verdict} confidence={result.confidence:.2f}"
    )
    print(f"reason={result.reason}")
    if not result.success:
        raise RuntimeError(result.error)


def load_config(path: str) -> dict:
    """读取统一config.yaml。"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """支持离线自测、真实API测试和相机AI复核三种入口。"""
    parser = argparse.ArgumentParser(description="豆包AI跌倒复核模块")
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
        action="store_true",
        help="使用环境变量密钥发送一次真实豆包请求",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.self_test:
        run_self_test(config)
    elif args.api_test:
        run_api_test(config)
    else:
        from main import run_live

        run_live(config, args.config, stage="ai")


if __name__ == "__main__":
    main()

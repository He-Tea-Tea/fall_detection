# -*- coding: utf-8 -*-
"""豆包视觉模型单图跌倒测试工具。

运行示例：
    python ai_test.py test.jpg
    python ai_test.py test.png --config config.yaml
    python ai_test.py "https://example.com/test.jpg"

返回结果：
    true      -> 检测到跌倒
    false     -> 未检测到跌倒
    uncertain -> 图片证据不足，无法确定

本文件只测试视觉API，不参与main.py的本地五维判断。
"""

import argparse
import base64
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import yaml
from openai import OpenAI


def load_config(config_path: str) -> dict:
    """读取指定的config.yaml并检查AI配置是否存在。"""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if "ai" not in config:
        raise KeyError("config.yaml中缺少ai配置段")

    return config


def get_api_key(config: dict) -> str:
    """优先读取config.yaml中的密钥，未填写时再读取环境变量。"""
    ai_config = config["ai"]

    config_key = str(ai_config.get("api_key", "")).strip()
    if config_key:
        return config_key

    environment_name = str(
        ai_config.get("api_key_env", "ARK_API_KEY")
    ).strip()
    environment_key = os.getenv(environment_name, "").strip()
    if environment_key:
        return environment_key

    raise RuntimeError(
        "未找到API Key，请填写config.yaml中的ai.api_key，"
        f"或设置环境变量{environment_name}"
    )


def validate_ai_config(config: dict) -> None:
    """在请求前检查必要配置，避免错误一直到API调用阶段才出现。"""
    ai_config = config["ai"]

    if not bool(ai_config.get("enabled", True)):
        raise RuntimeError("config.yaml中的ai.enabled为false")

    if not bool(ai_config.get("supports_vision", False)):
        raise RuntimeError(
            "当前测试需要上传图片，请把ai.supports_vision设为true"
        )

    base_url = str(ai_config.get("base_url", "")).strip()
    if not base_url.startswith("https://"):
        raise ValueError("ai.base_url必须是有效的HTTPS地址")

    if not str(ai_config.get("model", "")).strip():
        raise ValueError("ai.model不能为空")

    prompt_text = str(
        ai_config.get("prompt", {}).get("text", "")
    ).strip()
    if not prompt_text:
        raise ValueError("ai.prompt.text不能为空")

    request_config = ai_config.get("request")
    if not isinstance(request_config, dict):
        raise KeyError("config.yaml中缺少ai.request配置段")

    # 兼容新版max_output_tokens和旧版max_tokens。
    max_tokens = request_config.get(
        "max_output_tokens",
        request_config.get("max_tokens"),
    )
    if max_tokens is None or int(max_tokens) <= 0:
        raise ValueError(
            "请配置ai.request.max_output_tokens，且必须大于0"
        )


def local_image_to_data_url(
    image_path: str,
    image_config: dict,
) -> str:
    """读取本地图片，限制尺寸并压缩成JPEG Data URL。"""
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "OpenCV无法读取图片，请使用jpg、jpeg、png或webp格式"
        )

    # 图片过大时等比例缩小；小图片不会被放大。
    max_long_side = int(image_config.get("max_long_side_px", 768))
    if max_long_side <= 0:
        raise ValueError("ai.image.max_long_side_px必须大于0")

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

    jpeg_quality = int(image_config.get("jpeg_quality", 75))
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("ai.image.jpeg_quality必须在1到100之间")

    success, encoded_image = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not success:
        raise RuntimeError("图片JPEG编码失败")

    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("ascii")

    print(
        f"图片处理完成：{image.shape[1]}×{image.shape[0]}，"
        f"JPEG大小约{len(encoded_image) / 1024:.1f}KB"
    )

    return f"data:image/jpeg;base64,{image_base64}"


def prepare_image_url(image_input: str, config: dict) -> str:
    """把本地路径、图片URL或Data URL统一成API需要的image_url。"""
    image_input = str(image_input).strip()
    if not image_input:
        raise ValueError("图片输入不能为空")

    if image_input.startswith(("http://", "https://")):
        return image_input

    if image_input.startswith("data:image/"):
        return image_input

    image_config = config["ai"].get("image", {})
    return local_image_to_data_url(image_input, image_config)


def build_user_input(
    prompt_text: str,
    image_url: str,
) -> list:
    """构建Responses API输入，每次严格只包含一张图片。"""
    if not prompt_text.strip():
        raise ValueError("提示词不能为空")

    if not image_url.strip():
        raise ValueError("图片URL或Base64数据不能为空")

    content = [
        {
            "type": "input_image",
            "image_url": image_url,
        },
        {
            "type": "input_text",
            "text": prompt_text.strip(),
        },
    ]

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def extract_answer_text(response) -> str:
    """从Responses API返回对象中提取最终文本。"""
    # 新版OpenAI SDK通常可以直接使用output_text。
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # 兼容没有output_text快捷属性的返回格式。
    output = getattr(response, "output", None) or []
    for item in output:
        content_items = getattr(item, "content", None) or []

        for content_item in content_items:
            text = getattr(content_item, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

    raise ValueError("API响应中没有找到可解析的文本回答")


def parse_fall_answer(answer_text: str) -> Optional[bool]:
    """将true、false或uncertain转换成程序可以使用的结果。"""
    normalized = (
        answer_text.strip()
        .lower()
        .replace("```", "")
        .strip()
        .rstrip("。.!！")
    )

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    if normalized in {"uncertain", "unknown", "不确定"}:
        return None

    raise ValueError(
        f"AI回答格式无效：{answer_text!r}，"
        "模型必须只回答true、false或uncertain"
    )


def judge_fall(
    image_data: str,
    config: Optional[dict] = None,
) -> Tuple[Optional[bool], str]:
    """上传一张图片并判断是否跌倒。

    返回值：
        (True, 原始回答)  ：检测到跌倒。
        (False, 原始回答) ：明确未检测到跌倒。
        (None, 原始回答)  ：不确定、断网、超时或调用失败。
    """
    if config is None:
        default_config = Path(__file__).resolve().parent / "config.yaml"
        config = load_config(str(default_config))

    validate_ai_config(config)

    ai_config = config["ai"]
    request_config = ai_config["request"]

    base_url = str(ai_config["base_url"]).rstrip("/")
    model = str(ai_config["model"]).strip()
    prompt_text = str(ai_config["prompt"]["text"]).strip()
    api_key = get_api_key(config)

    image_url = prepare_image_url(image_data, config)
    user_input = build_user_input(prompt_text, image_url)

    max_output_tokens = int(
        request_config.get(
            "max_output_tokens",
            request_config.get("max_tokens"),
        )
    )
    timeout_s = float(request_config.get("timeout_s", 5.0))
    temperature = float(request_config.get("temperature", 0.0))
    top_p = float(request_config.get("top_p", 1.0))
    thinking_enabled = bool(
        request_config.get("thinking_enabled", False)
    )
    extra_headers = request_config.get("extra_headers", {}) or {}

    extra_body = {
        "thinking": {
            "type": "enabled" if thinking_enabled else "disabled"
        }
    }

    # max_retries=0可以避免断网时SDK自动重试造成长时间等待。
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
    )

    try:
        response = client.responses.create(
            model=model,
            input=user_input,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )

        answer_text = extract_answer_text(response)
        is_fall = parse_fall_answer(answer_text)
        return is_fall, answer_text

    except Exception as error:
        # 测试失败不能被误认为“明确没有跌倒”，所以返回None。
        print(f"AI调用失败：{type(error).__name__}: {error}")
        return None, ""


def main() -> None:
    """命令行单图测试入口。"""
    default_config = Path(__file__).resolve().parent / "config.yaml"

    parser = argparse.ArgumentParser(
        description="豆包视觉模型单图跌倒测试"
    )
    parser.add_argument(
        "image",
        help="本地图片路径、HTTP图片URL或Data URL",
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="config.yaml路径",
    )
    args = parser.parse_args()

    print(f"正在分析图片：{args.image}")

    try:
        config = load_config(args.config)
        is_fall, answer = judge_fall(args.image, config)
    except Exception as error:
        print(f"测试启动失败：{type(error).__name__}: {error}")
        return

    if is_fall is True:
        print("结果：是，检测到有人跌倒")
    elif is_fall is False:
        print("结果：否，未检测到有人跌倒")
    else:
        print("结果：不确定，AI无有效结论，不能当成未跌倒")

    print(f"AI原始回答：{answer or '<empty>'}")


if __name__ == "__main__":
    main()
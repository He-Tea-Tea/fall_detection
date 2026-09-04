import os
import yaml
import base64
from openai import OpenAI

# ---------- 读取 config.yaml ----------
def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ---------- 获取 API Key ----------
def get_api_key(config):
    yaml_key = config.get('ai', {}).get('api_key', '').strip()
    if yaml_key:
        return yaml_key
    env_key = os.getenv('ARK_API_KEY', '').strip()
    if env_key:
        return env_key
    raise RuntimeError("未找到有效 API Key，请检查 config.yaml 的 ai.api_key 或环境变量 ARK_API_KEY")

# ---------- 构建用户输入 ----------
def build_user_input(prompt_text, image_data=None):
    """
    构建 input 数组。
    image_data: 可以是图片URL、本地路径 或 base64编码数据（支持 data:image/...;base64,格式）
    如果 image_data 是本地路径，会自动转换为 base64
    """
    content = []
    
    # 处理图片
    if image_data:
        image_url = image_data
        # 如果是本地文件路径，转换为 base64
        if os.path.isfile(image_data):
            with open(image_data, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
                ext = os.path.splitext(image_data)[1].lower()
                mime_type = 'jpeg' if ext in ['.jpg', '.jpeg'] else 'png'
                image_url = f"data:image/{mime_type};base64,{img_base64}"
        content.append({"type": "input_image", "image_url": image_url})
    
    # 添加文本提示词（必须存在）
    if not prompt_text or not prompt_text.strip():
        raise ValueError("config.yaml 中 ai.prompt.text 不能为空")
    content.append({"type": "input_text", "text": prompt_text.strip()})
    
    return [{"role": "user", "content": content}]

# ---------- 核心函数：判断是否跌倒 ----------
def judge_fall(image_data, config=None):
    """
    判断图片中是否发生跌倒
    image_data: 图片路径 或 base64 数据
    返回: (bool, str) -> (是否跌倒, 原始回答)
    """
    if config is None:
        config = load_config()
    
    # ----- 从配置中读取所有必要参数（无默认值） -----
    ai_cfg = config.get('ai')
    if ai_cfg is None:
        raise KeyError("config.yaml 中缺少 'ai' 节")
    
    # 基础参数
    base_url = ai_cfg.get('base_url')
    if base_url is None:
        raise KeyError("config.yaml 中 ai.base_url 未配置")
    
    model = ai_cfg.get('model')
    if model is None:
        raise KeyError("config.yaml 中 ai.model 未配置")
    
    # request 子节参数
    req_cfg = ai_cfg.get('request')
    if req_cfg is None:
        raise KeyError("config.yaml 中 ai.request 未配置")
    
    max_tokens = req_cfg.get('max_tokens')
    if max_tokens is None:
        raise KeyError("config.yaml 中 ai.request.max_tokens 未配置")
    
    temperature = req_cfg.get('temperature')
    if temperature is None:
        raise KeyError("config.yaml 中 ai.request.temperature 未配置")
    
    # top_p 是可选的，但建议配置；如果缺失则使用 1.0（保守默认）
    top_p = req_cfg.get('top_p', 1.0)  # 提供默认值，因为部分模型可能不需要
    
    # thinking_enabled 是可选的，默认为 False（保守）
    thinking_enabled = req_cfg.get('thinking_enabled', False)
    
    # extra_headers 可选，默认为空字典
    extra_headers = req_cfg.get('extra_headers', {}) or {}
    
    # 提示词必须存在
    prompt_text = ai_cfg.get('prompt', {}).get('text')
    if prompt_text is None or not prompt_text.strip():
        raise ValueError("config.yaml 中 ai.prompt.text 未配置或为空")
    
    # ----- 获取 API Key -----
    api_key = get_api_key(config)
    
    # ----- 构建请求 -----
    user_input = build_user_input(prompt_text, image_data)
    
    extra_body = {'thinking': {'type': 'enabled' if thinking_enabled else 'disabled'}}
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    try:
        response = client.responses.create(
            model=model,
            input=user_input,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )
        
        # 提取回答文本
        answer_text = ""
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            answer_text = content_item.text.strip()
                            break
                if answer_text:
                    break
        
        # 统一转小写并去除首尾空格，然后判断是否等于 "true"
        is_fall = answer_text.strip().lower() == "true"
        return is_fall, answer_text
        
    except Exception as e:
        print(f"❌ AI调用失败：{e}")
        return False, ""

# ---------- 命令行测试入口 ----------
def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python ai_test.py <图片路径或URL>")
        print("示例: python ai_test.py test.jpg")
        return
    image_input = sys.argv[1]
    print(f"📷 正在分析图片: {image_input}")
    is_fall, answer = judge_fall(image_input)
    if is_fall:
        print("⚠️  结果：是 (检测到跌倒)")
    else:
        print("✅ 结果：否 (未检测到跌倒)")
    print(f"📝 AI原始回答: {answer}")

if __name__ == "__main__":
    main()
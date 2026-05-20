import json
import os
import sys
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


def _parse_model_aliases(raw: str | None) -> Dict[str, str]:
    """
    解析 MODEL_ALIASES 环境变量为 {client_model: upstream_model} 字典。

    支持两种格式：
    1. JSON 对象：{"claude-3-5-sonnet": "gpt-4o", "claude-3-haiku": "gpt-4o-mini"}
    2. 逗号分隔的键值对：claude-3-5-sonnet=gpt-4o,claude-3-haiku=gpt-4o-mini

    任意解析失败都会打印警告并返回空字典，确保服务能正常启动（无映射即透传）。
    """
    if not raw or not raw.strip():
        return {}

    raw = raw.strip()

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"WARNING: MODEL_ALIASES JSON 解析失败，将忽略别名映射: {e}")
            return {}
        if not isinstance(data, dict):
            print("WARNING: MODEL_ALIASES 必须是 JSON 对象，将忽略别名映射")
            return {}
        result: Dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                result[k.strip()] = v.strip()
            else:
                print(f"WARNING: MODEL_ALIASES 跳过无效条目: {k!r} -> {v!r}")
        return result

    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"WARNING: MODEL_ALIASES 跳过无效条目（缺少 '='）: {pair!r}")
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v:
            result[k] = v
        else:
            print(f"WARNING: MODEL_ALIASES 跳过空键或空值: {pair!r}")
    return result


class Settings:
    OPENAI_API_BASE: str
    OPENAI_API_KEY: str
    MODEL_ALIASES: Dict[str, str]

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            print("CRITICAL ERROR: OPENAI_API_KEY is missing or empty in environment variables.")
            print("Please set OPENAI_API_KEY in your .env file or environment.")
            sys.exit(1)
        self.OPENAI_API_KEY = api_key

        self.OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not self.OPENAI_API_BASE.endswith("/"):
            self.OPENAI_API_BASE += "/"

        self.TIMEOUT_CONNECT = float(os.getenv("TIMEOUT_CONNECT", "5.0"))
        self.TIMEOUT_READ = float(os.getenv("TIMEOUT_READ", "300.0"))
        self.TIMEOUT_WRITE = float(os.getenv("TIMEOUT_WRITE", "20.0"))
        self.TIMEOUT_POOL = float(os.getenv("TIMEOUT_POOL", "10.0"))

        self.MAX_BODY_SIZE = int(os.getenv("MAX_BODY_SIZE", "15728640"))

        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

        self.MODEL_ALIASES = _parse_model_aliases(os.getenv("MODEL_ALIASES"))


settings = Settings()

# ai-proxy：多协议 AI 代理

## 核心场景

你手里是**第三方 OpenAI 兼容接口**（`/v1/chat/completions`），本服务同时对外暴露 **Anthropic Messages API** 和 **OpenAI Chat Completions API**，满足不同客户端的接入需求。

### 暴露的端点


| 端点                          | 协议                          | 说明                                                     |
| --------------------------- | --------------------------- | ------------------------------------------------------ |
| `POST /v1/messages`         | Anthropic Messages API      | 接收 Anthropic 格式请求，转换为 OpenAI 格式转发上游，响应再转回 Anthropic 格式 |
| `POST /v1/chat/completions` | OpenAI Chat Completions API | 直接透传 OpenAI 格式请求到上游，原样返回响应                             |
| `GET /v1/models`            | OpenAI Models API           | 直接透传上游模型列表，原样返回                                        |


两个端点均支持**流式（SSE）和非流式**模式。代理不做模型名映射，客户端传什么 model 就原样转发。

## 客户端接入

### Claude Code / Anthropic SDK

把「Anthropic 的地址」指到本代理（注意路径里带 `/v1`）：

```bash
export ANTHROPIC_BASE_URL="http://<你的代理主机>:8000/v1"
export ANTHROPIC_API_KEY="dummy"
```

代理访问上游时用的是 `.env` 里的 `OPENAI_API_KEY`；这里的 `dummy` 仅满足 SDK「非空 Key」即可。

### OpenAI SDK / 其他 OpenAI 兼容客户端

直接将 Base URL 指向本代理：

```bash
export OPENAI_BASE_URL="http://<你的代理主机>:8000/v1"
export OPENAI_API_KEY="dummy"
```

或者在代码中：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<你的代理主机>:8000/v1",
    api_key="dummy",
)
resp = client.chat.completions.create(
    model="你的模型名",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## 环境变量


| 变量                | 含义                                                       |
| ----------------- | -------------------------------------------------------- |
| `OPENAI_API_BASE` | 第三方 OpenAI 兼容服务的 Base URL（配置里会自动补全末尾 `/`）                |
| `OPENAI_API_KEY`  | 上游要求的 API Key                                            |
| `LOG_LEVEL`       | `INFO`（默认）或 `DEBUG`；`DEBUG` 会打印上游每条 SSE 分片 JSON，流式时日志量很大 |


## 运行

```bash
uv sync
cp .env.example .env   # 再编辑 .env
uv run main.py
```

默认监听 `http://0.0.0.0:8000`。
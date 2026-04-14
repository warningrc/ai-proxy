# ai-proxy：对外 Anthropic，上游 OpenAI

## 核心场景

你手里是**第三方 OpenAI 兼容接口**（`/v1/chat/completions`），但 **Claude Code（CC）等工具只认 Anthropic 的 Messages API**（`/v1/messages`）。本服务做的是：

- **对外**：暴露与 Anthropic 兼容的 `POST /v1/messages`（含流式 SSE）。
- **对内**：把请求**转成 OpenAI Chat Completions**，转发到你配置的 `OPENAI_API_BASE`。

在 CC 里直接选用**上游已支持的 `model` 名称**即可；代理不做模型名映射。

## Claude Code / Anthropic SDK 怎么指过来

把「Anthropic 的地址」指到本代理（注意路径里带 `/v1`）：

```bash
export ANTHROPIC_BASE_URL="http://<你的代理主机>:8000/v1"
export ANTHROPIC_API_KEY="dummy"
```

代理访问上游时用的是 `.env` 里的 `OPENAI_API_KEY`；这里的 `dummy` 仅满足 SDK「非空 Key」即可。

## 环境变量

| 变量 | 含义 |
|------|------|
| `OPENAI_API_BASE` | 第三方 OpenAI 兼容服务的 Base URL（配置里会自动补全末尾 `/`） |
| `OPENAI_API_KEY` | 上游要求的 API Key |
| `LOG_LEVEL` | `INFO`（默认）或 `DEBUG`；`DEBUG` 会打印上游每条 SSE 分片 JSON，流式时日志量很大 |

## 运行

```bash
uv sync
cp .env.example .env   # 再编辑 .env
uv run main.py
```

默认监听 `http://0.0.0.0:8000`。

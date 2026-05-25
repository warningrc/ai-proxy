# ai-proxy：多协议 AI 代理

## 核心场景

你手里是**第三方 OpenAI 兼容接口**（`/v1/chat/completions`），本服务同时对外暴露 **Anthropic Messages API** 和 **OpenAI Chat Completions API**，满足不同客户端的接入需求。

### 暴露的端点


| 端点                          | 协议                          | 说明                                                     |
| --------------------------- | --------------------------- | ------------------------------------------------------ |
| `POST /v1/messages`         | Anthropic Messages API      | 接收 Anthropic 格式请求，转换为 OpenAI 格式转发上游，响应再转回 Anthropic 格式 |
| `POST /v1/chat/completions` | OpenAI Chat Completions API | 直接透传 OpenAI 格式请求到上游，原样返回响应                             |
| `GET /v1/models`            | OpenAI Models API           | 直接透传上游模型列表（默认 provider）                                   |


两个端点均支持**流式（SSE）和非流式**模式。支持配置**多个上游服务商**，按客户端传入的模型名路由到对应的 provider；同一个 provider 可声明对 OpenAI / Anthropic 两种协议的原生支持，`/v1/messages` 会自动选择最合适的转发模式——provider 有 Anthropic 端点 → 直接透传，否则走 claude→openai 协议转换（详见下方「多服务商与模型路由」）。


## 客户端接入

### Claude Code / Anthropic SDK

把「Anthropic 的地址」指到本代理（注意路径里带 `/v1`）：

```bash
export ANTHROPIC_BASE_URL="http://<你的代理主机>:8000/v1"
export ANTHROPIC_API_KEY="dummy"
```

代理访问上游时用的是 `config.toml` 里声明的 `api_key`；这里的 `dummy` 仅满足 SDK「非空 Key」即可。

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


## 配置

所有配置集中在项目根的 `config.toml` 中。**不再支持 `.env` 与环境变量覆盖**（除了 `CONFIG_FILE` 用于指定文件路径本身）。

首次启动前：

```bash
cp config.example.toml config.toml   # 编辑你的真实配置
```

`config.toml` 已加入 `.gitignore`，不会被提交。

### 顶层字段

| 字段                 | 类型     | 默认值        | 说明                                                                       |
| ------------------ | ------ | ---------- | ------------------------------------------------------------------------ |
| `log_level`        | string | `"INFO"`   | `"INFO"` 或 `"DEBUG"`；`"DEBUG"` 会打印上游每条 SSE 分片 JSON，流式时日志量很大                |
| `max_body_size`    | int    | `15728640` | 请求体字节上限（约 15 MiB）                                                          |
| `default_provider` | string | 第一个 provider 的 id | 未命中 `model_routes` 时使用的 provider id                                       |
| `[timeouts]`       | table  | —          | httpx 连接超时（秒）：`connect` / `read` / `write` / `pool`                       |
| `[[providers]]`    | array  | 至少一个       | 上游服务商列表，每项必须声明 `id` 与至少一个协议子表（`[providers.openai]` / `[providers.anthropic]`） |
| `[model_routes]`   | table  | 可空         | 客户端模型名 → `{ provider, model }`（`model` 省略即透传客户端模型名）                       |


## 多服务商与模型路由

代理支持同时挂多个上游服务商，请求按「客户端传入的模型名」路由到不同的 provider 与上游模型名。

### 基本概念

- `[[providers]]`：声明所有可用上游。顶层只放 `id` 与可选的**共享 `api_key`**；每个 provider 至少要有一个**协议子表**：
  - `[providers.openai]`    —— 该 provider 原生支持 OpenAI 协议
  - `[providers.anthropic]` —— 该 provider 原生支持 Anthropic Messages 协议

  两个协议的 `base_url` 必须各自写在子表里（根路径天然不同）。注意两个协议的拼接约定不一样：

  | 协议子表                 | `base_url` 是否包含 `/v1` | 代码拼接的相对路径              | 最终上游 URL（示例）                              |
  | ------------------------ | ------------------------- | ------------------------------- | ------------------------------------------------- |
  | `[providers.openai]`     | **包含** `/v1`            | `chat/completions` / `models`   | `https://api.openai.com/v1/chat/completions`     |
  | `[providers.anthropic]`  | **不包含** `/v1`          | `v1/messages`                   | `https://api.anthropic.com/v1/messages`          |

  这是各家官方 SDK 的事实约定（OpenAI 把 `v1` 视作 base 的一部分，Anthropic 把 `v1` 视作端点路径的一部分），混着写会出现 `/v1/v1/messages`。如果 anthropic 子表的 `base_url` 末尾带了 `/v1`，启动时会打 WARN 提示。

- `api_key` 的取值顺序：**子表 `api_key` > 顶层共享 `api_key`**。大部分三方服务两个协议用同一个 Key，只写顶层就够了；个别协议需要独立 Key 再在子表里写即可覆盖。两处都没有则启动失败。
- 鉴权头由子表的 `auth_style` 决定，可由子表 `headers` 进一步逐项覆盖：

  | auth_style    | 注入的默认鉴权头                                                |
  | ------------- | --------------------------------------------------------------- |
  | `"bearer"`    | `Authorization: Bearer <api_key>`                                |
  | `"anthropic"` | `x-api-key: <api_key>` + `anthropic-version: 2023-06-01`        |
  | `"none"`      | 不注入（自行用 `headers` 子表处理）                                    |

  `auth_style` 缺省时按协议默认派发：`[providers.openai]` → `"bearer"`、`[providers.anthropic]` → `"anthropic"`。某些厂商的 Anthropic 兼容端点其实要求 Bearer 鉴权（如火山方舟），这种时候在子表里显式写 `auth_style = "bearer"` 即可。

- `default_provider`：兜底 provider，未命中路由 / `/v1/models` 接口会用它。
- `[model_routes]`：把客户端发来的 `model` 字段路由到具体 provider，并可改写为上游真实模型名。
- 未命中路由的模型 → 默认 provider，模型名原样透传。

### `/v1/messages` 协议自动切换

| 路由命中的 provider | 实际转发模式 | 说明 |
| --- | --- | --- |
| 有 `[providers.anthropic]` 端点 | **anthropic-passthrough** | 直接发到 `{anthropic.base_url}/v1/messages`，body 仅替换 `model` 字段；客户端原始 query string（如 `?beta=true`）原样透传 |
| 仅有 `[providers.openai]` 端点 | **openai-convert** | 走 claude→openai 转换 → `{openai.base_url}/chat/completions` → openai→claude 反转（既有行为） |

`/v1/chat/completions` 仅会用到 provider 的 OpenAI 端点；如果路由到的 provider 缺 `[providers.openai]`，会返回 `502 { error.type = "provider_protocol_unsupported" }`。

### 配置示例

```toml
log_level        = "INFO"
default_provider = "openai-only"

# 只支持 OpenAI 协议
[[providers]]
id      = "openai-only"
api_key = "sk-xxx"

[providers.openai]
base_url = "https://api.openai.com/v1"
# api_key 继承顶层

# 同时支持两种协议，共享同一个 Key（大多数三方服务的情况）
[[providers]]
id      = "dual-shared"
api_key = "sk-vendor-xxx"

[providers.openai]
base_url = "https://api.example.com/openai/v1"

[providers.anthropic]
# Anthropic 协议 base_url 不要带 /v1；代码会自动拼 v1/messages
base_url = "https://api.example.com/anthropic"

# 两个协议要求各自独立 Key（少数情况）
[[providers]]
id = "dual-separate"

[providers.openai]
base_url = "https://api.example.com/openai/v1"
api_key  = "sk-openai-xxx"

[providers.anthropic]
base_url = "https://api.example.com/anthropic"
api_key  = "sk-ant-yyy"

[model_routes]
"claude-3-5-sonnet-20241022" = { provider = "openai-only", model = "gpt-4o" }
"claude-3-5-haiku-20241022"  = { provider = "openai-only", model = "gpt-4o-mini" }
"deepseek-chat"              = { provider = "dual-shared" }
```

要点：

- 路由 value 的 `model` 字段可省略 → 透传客户端传入的模型名（仅切换 provider）。
- 路由 value 也接受**数组表形式**（为未来的多 provider failover/loadbalance 预留），当前实现只使用 `list[0]`，存在多条时启动日志会出现 WARN。

  ```toml
  [[model_routes."claude-3-5-sonnet-failover"]]
  provider = "openai-only"
  model    = "gpt-4o"
  [[model_routes."claude-3-5-sonnet-failover"]]
  provider = "dual-protocol"
  model    = "gpt-4o"
  ```

### 命中日志

每次命中路由时会写一条 INFO；`/v1/messages` 还会额外打出当前走的是哪个 mode：

```
[abc123def456] model route | 'claude-3-5-sonnet-20241022' -> provider='dual-protocol' model='gpt-4o'
[abc123def456] /v1/messages | provider='dual-protocol' | mode=anthropic-passthrough | model='gpt-4o' | stream=true
```

启动时也会打印当前 providers 与各协议 `base_url`、routes 的概要，方便确认配置是否被加载。


## 运行

```bash
uv sync
cp config.example.toml config.toml   # 编辑你的真实配置
uv run main.py
```

默认监听 `http://0.0.0.0:8000`。如需把配置文件放到其他位置：

```bash
CONFIG_FILE=/etc/ai-proxy/config.toml uv run main.py
```

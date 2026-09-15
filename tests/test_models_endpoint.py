"""覆盖 /v1/models 的条目构造、命名空间、排序与路由解析（纯函数部分）。"""

import asyncio

import pytest

from config import ModelRoute, ProviderConfig, _parse_providers, settings
import main
from main import (
    _models_payload,
    _normalize_upstream_models,
    _route_model_entries,
    discover_upstream_models,
    resolve_model_route,
)


def _routes(**name_to_route):
    """便捷构造 Dict[str, List[ModelRoute]]。"""
    return {
        name: [ModelRoute(provider_id=provider, upstream_model=upstream)]
        for name, (provider, upstream) in name_to_route.items()
    }


def test_route_model_entries_tags_and_passthrough():
    """model_routes 条目带标记；省略 upstream 时透传客户端模型名。"""
    routes = _routes(
        **{
            "claude-3-5-sonnet-20241022": ("openai-only", "gpt-4o"),
            "deepseek-chat": ("dual-shared-key", None),
        }
    )
    entries = _route_model_entries(routes, "openai-only")

    assert entries["claude-3-5-sonnet-20241022"] == {
        "id": "claude-3-5-sonnet-20241022",
        "object": "model",
        "source": "model_routes",
        "provider": "openai-only",
        "upstream_model": "gpt-4o",
    }
    assert entries["deepseek-chat"]["provider"] == "dual-shared-key"
    assert entries["deepseek-chat"]["source"] == "model_routes"
    assert entries["deepseek-chat"]["upstream_model"] == "deepseek-chat"


def test_normalize_upstream_models_namespaces_and_marks():
    """上游模型统一加 provider 前缀并打标记，非法条目被忽略。"""
    upstream = [
        {"id": "gpt-4o", "object": "model", "created": 1, "owned_by": "openai"},
        "not-a-dict",
        {"object": "model"},  # 缺 id
        {"id": ""},  # 空 id
        {"id": "  deepseek-chat  "},  # 带空格，规范化
    ]
    entries = _normalize_upstream_models("openai-only", upstream)

    assert set(entries) == {"openai-only/gpt-4o", "openai-only/deepseek-chat"}
    assert entries["openai-only/gpt-4o"] == {
        "id": "openai-only/gpt-4o",
        "object": "model",
        "created": 1,
        "owned_by": "openai",
        "source": "upstream",
        "provider": "openai-only",
        "upstream_model": "gpt-4o",
        "models_source": "api",
    }
    assert entries["openai-only/deepseek-chat"]["upstream_model"] == "deepseek-chat"


def test_normalize_upstream_models_manual_source():
    """手动模型列表生成的条目带 models_source="manual"。"""
    entries = _normalize_upstream_models(
        "volcengine", [{"id": "deepseek-v4-pro"}], models_source="manual",
    )
    assert entries["volcengine/deepseek-v4-pro"]["models_source"] == "manual"
    assert entries["volcengine/deepseek-v4-pro"]["source"] == "upstream"


def test_models_payload_routes_first_then_upstream_sorted():
    """model_routes 保持配置顺序排最前，上游模型再按名称排序。"""
    routes = _routes(
        **{
            "Zulu": ("p", "z-upstream"),
            "alpha": ("p", "a-upstream"),
        }
    )
    merged = {
        **_normalize_upstream_models("p", [{"id": "Bravo"}, {"id": "delta"}]),
        **_route_model_entries(routes, "p"),
    }
    payload = _models_payload(merged, routes)
    assert payload["object"] == "list"
    assert [m["id"] for m in payload["data"]] == ["Zulu", "alpha", "p/Bravo", "p/delta"]


def test_models_payload_filters_missing_routes():
    """传入的 route_names 中不在合并结果里的项应被忽略。"""
    routes = _routes(**{"a": ("p", None)})
    merged = {
        **_route_model_entries(routes, "p"),
        **_normalize_upstream_models("p", [{"id": "b"}]),
    }
    payload = _models_payload(merged, ["a", "not-present"])
    assert [m["id"] for m in payload["data"]] == ["a", "p/b"]


def test_resolve_model_route_namespaced(monkeypatch):
    """命名空间 "provider/model" 解析到对应 provider，未知前缀走兜底。"""
    monkeypatch.setattr(settings, "MODEL_ROUTES", {}, raising=False)
    monkeypatch.setattr(settings, "PROVIDERS", {"p1": object(), "p2": object()}, raising=False)
    monkeypatch.setattr(settings, "DEFAULT_PROVIDER_ID", "p1", raising=False)

    assert resolve_model_route("p2/gpt-4o", "r") == ("p2", "gpt-4o")
    # 未知前缀 → 兜底 default provider，模型名原样透传
    assert resolve_model_route("unknown/foo", "r") == ("p1", "unknown/foo")
    # 无斜杠 → 兜底
    assert resolve_model_route("gpt-4o", "r") == ("p1", "gpt-4o")


def test_resolve_model_route_prefers_static_routes(monkeypatch):
    """model_routes 显式配置优先于命名空间解析。"""
    monkeypatch.setattr(
        settings,
        "MODEL_ROUTES",
        {"p1/alias": [ModelRoute(provider_id="p2", upstream_model="real")]},
        raising=False,
    )
    monkeypatch.setattr(settings, "PROVIDERS", {"p1": object(), "p2": object()}, raising=False)
    monkeypatch.setattr(settings, "DEFAULT_PROVIDER_ID", "p1", raising=False)

    assert resolve_model_route("p1/alias", "r") == ("p2", "real")


def test_slash_in_model_name_is_preserved(monkeypatch):
    """模型名本身含 '/'（如 MiniMax/M3）时，按第一个 '/' 切分，剩余部分完整保留。"""
    monkeypatch.setattr(settings, "MODEL_ROUTES", {}, raising=False)
    monkeypatch.setattr(settings, "PROVIDERS", {"minimax": object()}, raising=False)
    monkeypatch.setattr(settings, "DEFAULT_PROVIDER_ID", "minimax", raising=False)

    entries = _normalize_upstream_models("minimax", [{"id": "MiniMax/M3"}])
    assert set(entries) == {"minimax/MiniMax/M3"}
    assert entries["minimax/MiniMax/M3"]["upstream_model"] == "MiniMax/M3"

    # 命名空间 "provider/MiniMax/M3" → provider + "MiniMax/M3"
    assert resolve_model_route("minimax/MiniMax/M3", "r") == ("minimax", "MiniMax/M3")


def test_provider_id_cannot_contain_slash():
    """provider id 不能含 '/'，否则会破坏命名空间路由的切分。"""
    with pytest.raises(SystemExit):
        _parse_providers([{"id": "bad/id"}])


def test_discovery_prefers_manual_models(monkeypatch):
    """provider 配置了 models 手动列表时直接采用，不请求上游 /models。"""
    main._invalidate_upstream_models_cache()
    monkeypatch.setattr(main, "http_clients", {}, raising=False)
    monkeypatch.setattr(
        settings, "PROVIDERS",
        {
            "volcengine": ProviderConfig(
                id="volcengine", openai=None, anthropic=None,
                models=["deepseek-v4-pro", "MiniMax/M3", "deepseek-v4-pro"],
            ),
            "no-models": ProviderConfig(id="no-models"),
        },
        raising=False,
    )

    try:
        discovered = asyncio.run(discover_upstream_models(force=True))
        # 重复项去重；含 '/' 的模型名完整保留
        assert set(discovered) == {"volcengine/deepseek-v4-pro", "volcengine/MiniMax/M3"}
        assert discovered["volcengine/MiniMax/M3"]["upstream_model"] == "MiniMax/M3"
        assert discovered["volcengine/deepseek-v4-pro"]["models_source"] == "manual"
    finally:
        main._invalidate_upstream_models_cache()

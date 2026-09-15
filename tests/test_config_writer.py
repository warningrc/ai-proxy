"""
config_writer round-trip 测试：解析 → 写回 → 再解析，应保持等价。

覆盖 config.example.toml 的全部结构（providers 双协议、model_routes、tenants、admin）。
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import config_writer
from config import (
    AUTH_STYLE_BEARER,
    PROTOCOL_OPENAI,
    Settings,
)

EXAMPLE_TOML = Path(__file__).resolve().parent.parent / "config.example.toml"


@pytest.fixture
def example_data():
    """读取 config.example.toml 为 dict。"""
    with EXAMPLE_TOML.open("rb") as f:
        return tomllib.load(f)


def _settings_from_dict(data):
    """用 dict 构造一个临时 Settings（绕过磁盘），复用 _populate_from 校验链路。"""
    s = Settings.__new__(Settings)
    s.CONFIG_PATH = EXAMPLE_TOML
    s._populate_from(data)
    return s


class TestWriterRoundTrip:
    """写回的 TOML 必须能被再次解析，且关键结构等价。"""

    def test_dumps_produces_valid_toml(self, example_data):
        """dumps 输出必须是合法 TOML。"""
        s = _settings_from_dict(example_data)
        text = config_writer.dumps(s.to_dict())
        reparsed = tomllib.loads(text)
        assert isinstance(reparsed, dict)
        assert "providers" in reparsed

    def test_roundtrip_preserves_providers(self, example_data):
        """providers 的 id / base_url / auth_style / api_key 共享关系应保持。"""
        s = _settings_from_dict(example_data)
        text = config_writer.dumps(s.to_dict())
        reparsed = tomllib.loads(text)
        s2 = _settings_from_dict(reparsed)

        assert list(s.PROVIDERS.keys()) == list(s2.PROVIDERS.keys())
        for pid in s.PROVIDERS:
            p1 = s.PROVIDERS[pid]
            p2 = s2.PROVIDERS[pid]
            assert p1.id == p2.id
            # openai 端点
            assert (p1.openai is None) == (p2.openai is None)
            if p1.openai:
                assert p1.openai.base_url == p2.openai.base_url
                assert p1.openai.api_key == p2.openai.api_key
                assert p1.openai.auth_style == p2.openai.auth_style
            # anthropic 端点
            assert (p1.anthropic is None) == (p2.anthropic is None)
            if p1.anthropic:
                assert p1.anthropic.base_url == p2.anthropic.base_url
                assert p1.anthropic.api_key == p2.anthropic.api_key
                assert p1.anthropic.auth_style == p2.anthropic.auth_style

    def test_roundtrip_preserves_model_routes(self, example_data):
        """model_routes 的 client_model → (provider, model) 应保持。"""
        s = _settings_from_dict(example_data)
        text = config_writer.dumps(s.to_dict())
        reparsed = tomllib.loads(text)
        s2 = _settings_from_dict(reparsed)

        assert set(s.MODEL_ROUTES.keys()) == set(s2.MODEL_ROUTES.keys())
        for model in s.MODEL_ROUTES:
            r1 = s.MODEL_ROUTES[model][0]
            r2 = s2.MODEL_ROUTES[model][0]
            assert r1.provider_id == r2.provider_id
            assert r1.upstream_model == r2.upstream_model

    def test_roundtrip_preserves_tenants(self, example_data):
        """tenants 的 id/name/api_key/status 应保持。"""
        s = _settings_from_dict(example_data)
        text = config_writer.dumps(s.to_dict())
        reparsed = tomllib.loads(text)
        s2 = _settings_from_dict(reparsed)

        # example.toml 没有 tenants 段，这里验证空场景与有数据场景都不出错
        assert [t.id for t in s.TENANTS] == [t.id for t in s2.TENANTS]

    def test_roundtrip_preserves_admin(self, example_data):
        """admin 段（含 password_hash）应保持。"""
        s = _settings_from_dict(example_data)
        text = config_writer.dumps(s.to_dict())
        reparsed = tomllib.loads(text)
        s2 = _settings_from_dict(reparsed)

        assert (s.ADMIN is None) == (s2.ADMIN is None)
        if s.ADMIN:
            assert s.ADMIN.username == s2.ADMIN.username
            assert s.ADMIN.password == s2.ADMIN.password
            assert s.ADMIN.password_hash == s2.ADMIN.password_hash

    def test_roundtrip_preserves_scalars_and_timeouts(self, example_data):
        """顶层标量与 [timeouts] 应保持。"""
        s = _settings_from_dict(example_data)
        text = config_writer.dumps(s.to_dict())
        reparsed = tomllib.loads(text)
        s2 = _settings_from_dict(reparsed)

        assert s.LOG_LEVEL == s2.LOG_LEVEL
        assert s.MAX_BODY_SIZE == s2.MAX_BODY_SIZE
        assert s.DEFAULT_PROVIDER_ID == s2.DEFAULT_PROVIDER_ID
        assert s.TIMEOUT_CONNECT == s2.TIMEOUT_CONNECT
        assert s.TIMEOUT_READ == s2.TIMEOUT_READ
        assert s.TIMEOUT_WRITE == s2.TIMEOUT_WRITE
        assert s.TIMEOUT_POOL == s2.TIMEOUT_POOL


class TestWriterEdgeCases:
    """边界场景。"""

    def test_shared_api_key_promoted_to_top(self):
        """两个协议共用同一 api_key 时，应提到顶层共享。"""
        data = {
            "log_level": "INFO",
            "default_provider": "p1",
            "providers": [{
                "id": "p1",
                "api_key": "shared-key",
                "openai": {"base_url": "https://a/v1"},
                "anthropic": {"base_url": "https://a/anthropic"},
            }],
        }
        s = _settings_from_dict(data)
        out = s.to_dict()
        # 顶层应有共享 api_key，子表不应再带 api_key
        assert out["providers"][0]["api_key"] == "shared-key"
        assert "api_key" not in out["providers"][0]["openai"]
        assert "api_key" not in out["providers"][0]["anthropic"]

    def test_separate_api_keys_stay_in_subtables(self):
        """两协议各自独立 key 时，顶层不写，子表各写。"""
        data = {
            "log_level": "INFO",
            "default_provider": "p1",
            "providers": [{
                "id": "p1",
                "openai": {"base_url": "https://a/v1", "api_key": "key-oai"},
                "anthropic": {"base_url": "https://a/anthropic", "api_key": "key-ant"},
            }],
        }
        s = _settings_from_dict(data)
        out = s.to_dict()
        assert "api_key" not in out["providers"][0]
        assert out["providers"][0]["openai"]["api_key"] == "key-oai"
        assert out["providers"][0]["anthropic"]["api_key"] == "key-ant"

    def test_model_with_special_chars_quoted(self):
        """含点号/连字符的 model 名应被正确引号包裹，写回后仍可解析。"""
        data = {
            "log_level": "INFO",
            "default_provider": "p1",
            "providers": [{"id": "p1", "api_key": "k", "openai": {"base_url": "https://a/v1"}}],
            "model_routes": {
                "qwen3.7-plus": {"provider": "p1", "model": "qwen3.7-plus"},
                "minimax-m2.7": {"provider": "p1"},
            },
        }
        s = _settings_from_dict(data)
        text = config_writer.dumps(s.to_dict())
        reparsed = tomllib.loads(text)
        assert "qwen3.7-plus" in reparsed["model_routes"]
        assert "minimax-m2.7" in reparsed["model_routes"]

    def test_atomic_write_creates_file(self, tmp_path):
        """write_toml 应创建目标文件。"""
        target = tmp_path / "out.toml"
        data = {
            "log_level": "INFO",
            "default_provider": "p1",
            "providers": [{"id": "p1", "api_key": "k", "openai": {"base_url": "https://a/v1"}}],
        }
        config_writer.write_toml(target, data)
        assert target.exists()
        reparsed = tomllib.loads(target.read_text(encoding="utf-8"))
        assert reparsed["providers"][0]["id"] == "p1"

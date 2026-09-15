"""
管理配置的脱敏与还原：GET 出去的密钥不能是明文，PUT 回来的掩码不能写进文件。

脱敏是「显示归显示、存储归存储」的约定：GET 把密钥换成掩码，界面原样回传掩码
时 PUT 再从当前 settings 还原成真值。两头必须成对 —— 只脱敏不还原，界面一保存
就把 "***" 写进 config.toml，所有 key 当场报废。
"""

from __future__ import annotations

import json

import pytest

import admin_api


class _FakeSettings:
    def __init__(self, cfg):
        self._cfg = cfg

    def to_dict(self):
        return json.loads(json.dumps(self._cfg))  # 深拷贝，别让测试互相污染


def _hash():
    return "a" * 64


@pytest.fixture
def real_config():
    return {
        "providers": [
            {
                "id": "p1",
                "api_key": "sk-real-shared-key-0001",
                "openai": {"base_url": "http://x/v1/", "api_key": "sk-real-openai-0001"},
                "anthropic": {"base_url": "http://x/", "api_key": "sk-real-anth-0001"},
            }
        ],
        "tenants": [{"id": "t1", "name": "n", "api_key": "sk-real-tenant-0001"}],
        "admin": {"username": "admin", "password_hash": _hash()},
    }


# ---------------------------------------------------------------------------
#  GET：脱敏
# ---------------------------------------------------------------------------

def test_password_hash_is_masked(real_config):
    """password_hash 必须和 api_key 一样脱敏。

    它不是「不可逆的摘要」而是管理会话 token 的签名密钥原料（见
    admin_auth._signing_key）：泄漏它 = 任何人可离线伪造管理员 token。
    """
    out = admin_api._mask_config(real_config)
    assert out["admin"]["password_hash"] != _hash()
    assert "***" in out["admin"]["password_hash"]


def test_no_secret_survives_masking(real_config):
    """把整个响应体拼成字符串，任何真 key 都不该出现在里面。"""
    out = admin_api._mask_config(real_config)
    blob = json.dumps(out)
    for secret in (
        "sk-real-shared-key-0001",
        "sk-real-openai-0001",
        "sk-real-anth-0001",
        "sk-real-tenant-0001",
        _hash(),
    ):
        assert secret not in blob


def test_masking_does_not_mutate_the_source(real_config):
    """脱敏只能作用于副本：顺手改了 settings 里的对象 = 线上配置被毁。"""
    admin_api._mask_config(real_config)
    assert real_config["providers"][0]["api_key"] == "sk-real-shared-key-0001"
    assert real_config["admin"]["password_hash"] == _hash()


def test_plaintext_password_is_also_masked():
    out = admin_api._mask_config(
        {"admin": {"username": "admin", "password": "plain-pw-123456"}}
    )
    assert "plain-pw-123456" not in json.dumps(out)


# ---------------------------------------------------------------------------
#  PUT：掩码还原
# ---------------------------------------------------------------------------

def test_masked_password_hash_is_restored(monkeypatch, real_config):
    """界面原样回传掩码 → 还原成真值，绝不能把 "***" 写进 config.toml。"""
    monkeypatch.setattr(admin_api, "settings", _FakeSettings(real_config))
    incoming = admin_api._mask_config(real_config)
    saved = admin_api._resolve_masked_keys(incoming)
    assert saved["admin"]["password_hash"] == _hash()


def test_mask_then_save_round_trips_every_key(monkeypatch, real_config):
    """GET → 原样 PUT 的往返必须是无损的（这是界面「不改就保存」的路径）。"""
    monkeypatch.setattr(admin_api, "settings", _FakeSettings(real_config))
    saved = admin_api._resolve_masked_keys(admin_api._mask_config(real_config))
    assert saved == real_config


def test_masked_hash_is_dropped_when_config_has_none(monkeypatch):
    """当前配置没有 password_hash（用的是明文 password）时，掩码要丢掉而不是写进去。"""
    cfg = {"admin": {"username": "admin", "password": "pw-123456"}}
    monkeypatch.setattr(admin_api, "settings", _FakeSettings(cfg))
    incoming = {"admin": {"username": "admin", "password_hash": "***"}}
    saved = admin_api._resolve_masked_keys(incoming)
    assert "password_hash" not in saved["admin"]


def test_new_plaintext_password_replaces_the_hash(monkeypatch, real_config):
    """用户在界面上输入新明文密码：要原样采用，并把旧的 password_hash 带过去由
    writer 覆盖掉 —— 但绝不能把掩码当成新密码。"""
    monkeypatch.setattr(admin_api, "settings", _FakeSettings(real_config))
    incoming = {
        "admin": {"username": "admin", "password": "brand-new-pw", "password_hash": "***"}
    }
    saved = admin_api._resolve_masked_keys(incoming)
    assert saved["admin"]["password"] == "brand-new-pw"
    assert saved["admin"]["password_hash"] == _hash()


def test_unmasked_input_is_taken_verbatim(monkeypatch, real_config):
    """用户输入的明文 key 原样采用（掩码判定只认 "***"）。"""
    monkeypatch.setattr(admin_api, "settings", _FakeSettings(real_config))
    incoming = {
        "providers": [{"id": "p1", "api_key": "sk-brand-new"}],
        "tenants": [{"id": "t1", "api_key": "sk-new-tenant"}],
    }
    saved = admin_api._resolve_masked_keys(incoming)
    assert saved["providers"][0]["api_key"] == "sk-brand-new"
    assert saved["tenants"][0]["api_key"] == "sk-new-tenant"

"""
admin_auth 测试：登录、token 颁发与校验、过期、未配置 admin。

为避免污染全局 settings，测试用 monkeypatch 临时替换 admin_auth.settings。
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

import admin_auth
from config import AdminConfig


class _FakeSettings:
    """用于测试的轻量 settings 替身。"""
    def __init__(self, admin=None):
        self.ADMIN = admin


def _set_admin(monkeypatch, admin):
    """临时替换 admin_auth 模块里的 settings。"""
    monkeypatch.setattr(admin_auth, "settings", _FakeSettings(admin=admin))


@pytest.fixture(autouse=True)
def _clean_revocation():
    """撤销名单是模块级全局，测试之间必须清干净，否则互相污染。"""
    admin_auth._revoked.clear()
    yield
    admin_auth._revoked.clear()


class TestIsAdminEnabled:
    def test_no_admin_section_disabled(self, monkeypatch):
        _set_admin(monkeypatch, admin=None)
        assert admin_auth.is_admin_enabled() is False

    def test_with_admin_section_enabled(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        assert admin_auth.is_admin_enabled() is True


class TestLoginPlaintext:
    def test_correct_credentials(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="secret"))
        token = admin_auth.login("admin", "secret")
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 20

    def test_wrong_password(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="secret"))
        assert admin_auth.login("admin", "wrong") is None

    def test_wrong_username(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="secret"))
        assert admin_auth.login("other", "secret") is None

    def test_admin_not_configured(self, monkeypatch):
        _set_admin(monkeypatch, admin=None)
        assert admin_auth.login("admin", "secret") is None


class TestLoginHash:
    def test_correct_credentials_hash(self, monkeypatch):
        # sha256("secret")
        import hashlib
        h = hashlib.sha256(b"secret").hexdigest()
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password_hash=h))
        token = admin_auth.login("admin", "secret")
        assert token is not None

    def test_wrong_password_hash(self, monkeypatch):
        import hashlib
        h = hashlib.sha256(b"secret").hexdigest()
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password_hash=h))
        assert admin_auth.login("admin", "wrong") is None

    def test_hash_preferred_over_password(self, monkeypatch):
        """同时配置时优先用 hash。"""
        import hashlib
        h = hashlib.sha256(b"hash-pw").hexdigest()
        _set_admin(monkeypatch, admin=AdminConfig(
            username="admin", password="plain-pw", password_hash=h
        ))
        # hash 校验通过
        assert admin_auth.login("admin", "hash-pw") is not None
        # 明文不应通过（因为优先 hash）
        assert admin_auth.login("admin", "plain-pw") is None


class TestTokenValidation:
    def test_valid_token(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        assert admin_auth.validate_token(token) == "admin"

    def test_invalid_token(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        assert admin_auth.validate_token("nonexistent-token") is None

    def test_empty_token(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        assert admin_auth.validate_token("") is None
        assert admin_auth.validate_token(None) is None

    def test_expired_token_rejected(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        # 过期时间写在 token 里，所以直接把 TTL 设成负数就得到一个"签发即过期"的 token
        monkeypatch.setattr(admin_auth, "TOKEN_TTL_SECONDS", -1)
        token = admin_auth.login("admin", "pw")
        assert admin_auth.validate_token(token) is None

    def test_revoke_token(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        admin_auth.revoke_token(token)
        assert admin_auth.validate_token(token) is None

    def test_revoke_nonexistent_token_silent(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        # 不应抛异常
        admin_auth.revoke_token("nonexistent")
        admin_auth.revoke_token(None)


class TestBase64CanonicalForm:
    """base64 的等价写法不能变成绕过手段。

    base64 编码 32 字节的签名时，末字符只有高 4 位有效、低 2 位是冗余位，
    "AA" 和 "AB"（乃至 "AC"/"AD"）解出来都是 b'\\x00'。所以同一个 token 有很多种
    字符串写法，只要校验或记录是按**字符串**做的，换个尾字符就能让「签名校验通过」
    和「撤销名单里查不到」同时成立 —— 登出形同虚设。
    """

    @staticmethod
    def _variant(token: str) -> str:
        """把签名段末尾换成「解出同样字节」的另一个 base64 字符。"""
        alphabet = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz0123456789-_")
        payload_b64, _, sig_b64 = token.partition(".")
        # 末字符的低 2 位是冗余位（规范编码里恒为 0），翻转最低位即可
        return f"{payload_b64}.{sig_b64[:-1]}{alphabet[alphabet.index(sig_b64[-1]) ^ 1]}"

    def test_variant_decodes_to_the_same_signature(self, monkeypatch):
        """先确认漏洞成立的前提：**宽松**解码器（base64 标准库的默认行为）看到的
        两个变体是同一串字节 —— 所以签名对两者都成立。_b64d 现在会直接拒掉变体，
        所以这里必须绕开它、用裸的 urlsafe_b64decode 来演示。"""
        import base64

        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        variant = self._variant(token)

        assert variant != token
        _, _, sig_b64 = token.partition(".")
        _, _, var_sig_b64 = variant.partition(".")
        assert base64.urlsafe_b64decode(sig_b64 + "==") == \
            base64.urlsafe_b64decode(var_sig_b64 + "==")

    def test_variant_token_is_rejected(self, monkeypatch):
        """非规范编码一律拒绝：只接受一种写法，等价写法就无从下手。"""
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        assert admin_auth.validate_token(self._variant(token)) is None

    def test_revocation_cannot_be_bypassed_with_a_variant(self, monkeypatch):
        """登出后换个等价写法照样要失效。"""
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        variant = self._variant(token)

        admin_auth.revoke_token(token)
        assert admin_auth.validate_token(token) is None
        assert admin_auth.validate_token(variant) is None

    def test_revocation_key_is_derived_from_decoded_bytes(self, monkeypatch):
        """撤销名单的键取自解码后的字节，所以等价写法天然共用同一个键。"""
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        payload_b64, _, sig_b64 = token.partition(".")
        payload = admin_auth._b64d(payload_b64)
        sig = admin_auth._b64d(sig_b64)

        admin_auth.revoke_token(token)
        assert admin_auth._token_key(payload, sig) in admin_auth._revoked

    @pytest.mark.parametrize(
        "bad",
        [
            "AB",      # 与 "AA" 解出同样的字节：等价写法的典型形态
            "AB==",    # 同上，且带上了签发端从不产生的 padding
            "AA==",    # 仅 padding 不合法
            "+/+/",    # 标准字母表写法，urlsafe 规范形式是 "-_-_"
        ],
    )
    def test_b64d_rejects_non_canonical(self, bad):
        with pytest.raises(ValueError):
            admin_auth._b64d(bad)

    @pytest.mark.parametrize("good", ["AA", "abcd", "a-b_"])
    def test_b64d_accepts_canonical(self, good):
        assert admin_auth._b64e(admin_auth._b64d(good)) == good

    def test_b64d_accepts_what_it_emits(self):
        import os
        raw = os.urandom(32)
        assert admin_auth._b64d(admin_auth._b64e(raw)) == raw


class TestPasswordHashIsTheSigningKey:
    """记录 password_hash 为什么必须脱敏（见 admin_api._mask_config）。

    它不是「不可逆的摘要」，而是管理会话 token 的签名密钥原料。拿到它就能离线
    伪造管理员 token —— 比拿到明文密码还省事，不用爆破。
    """

    def test_forged_token_from_leaked_hash(self, monkeypatch):
        import hashlib
        import hmac
        import json

        from usage_stats import _hash_key

        leaked_hash = _hash_key("pw")
        _set_admin(
            monkeypatch,
            admin=AdminConfig(username="admin", password_hash=leaked_hash),
        )

        key = hashlib.sha256(
            b"ai-proxy/admin-session/v1\x00" + leaked_hash.encode("utf-8")
        ).digest()
        payload = json.dumps(
            {"u": "admin", "e": time.time() + 3600},
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        sig = hmac.new(key, payload, hashlib.sha256).digest()
        forged = f"{admin_auth._b64e(payload)}.{admin_auth._b64e(sig)}"

        # 攻击者不需要知道密码原文，也不需要登录接口
        assert admin_auth.validate_token(forged) == "admin"


class TestStatelessToken:
    """多 worker 契约：token 只依赖配置里的凭据，不依赖任何进程内状态。

    之前 token 存在内存 dict 里，uvicorn --workers N 下登录落在 A、下一个请求
    落到 B 就 401，管理界面随机失效。
    """

    def test_token_is_self_contained(self, monkeypatch):
        """不查任何服务端状态也能校验通过（等价于另一个 worker 来校验）。"""
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        # 唯一的内存态就是撤销名单，空名单 = 全新 worker 的内存视角
        assert not admin_auth._revoked
        assert admin_auth.validate_token(token) == "admin"
        assert admin_auth.validate_token(token) == "admin"

    def test_password_change_invalidates_old_tokens(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="new-pw"))
        assert admin_auth.validate_token(token) is None

    def test_password_hash_also_works_across_workers(self, monkeypatch):
        from usage_stats import _hash_key

        _set_admin(
            monkeypatch,
            admin=AdminConfig(username="admin", password_hash=_hash_key("pw")),
        )
        token = admin_auth.login("admin", "pw")
        assert admin_auth.validate_token(token) == "admin"

    def test_tampered_payload_rejected(self, monkeypatch):
        """改 payload 里的用户名但不重签名 —— 必须拒绝。"""
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        payload_b64, _, sig_b64 = token.partition(".")
        payload = admin_auth._b64d(payload_b64)
        assert b'"admin"' in payload
        forged = payload.replace(b'"admin"', b'"root" ')
        forged_token = f"{admin_auth._b64e(forged)}.{sig_b64}"
        assert admin_auth.validate_token(forged_token) is None

    def test_tampered_signature_rejected(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        payload_b64, _, sig_b64 = token.partition(".")
        flipped = ("A" if sig_b64[0] != "A" else "B") + sig_b64[1:]
        assert admin_auth.validate_token(f"{payload_b64}.{flipped}") is None

    @pytest.mark.parametrize(
        "bad",
        [
            "abc",                    # 没有分隔符
            "a.b",                    # 两段都不是合法 base64 载荷
            "!!!.!!!",                # 非法 base64
            "....",                   # 一堆分隔符
            "x" * 5000,               # 超长
            "数据.签名",               # 非 ASCII
            "",                       # 空串
        ],
    )
    def test_garbage_token_never_raises(self, monkeypatch, bad):
        """token 是攻击者可控输入，校验过程不能抛异常。"""
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        assert admin_auth.validate_token(bad) is None


class TestRequireAdminDependency:
    def _make_request(self, headers=None):
        class _Req:
            def __init__(self, h):
                self.headers = h or {}
        return _Req(headers or {})

    def test_not_configured_returns_503(self, monkeypatch):
        _set_admin(monkeypatch, admin=None)
        req = self._make_request({"x-admin-token": "anything"})
        with pytest.raises(HTTPException) as exc:
            admin_auth.require_admin(req)
        assert exc.value.status_code == 503

    def test_missing_token_returns_401(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        req = self._make_request({})
        with pytest.raises(HTTPException) as exc:
            admin_auth.require_admin(req)
        assert exc.value.status_code == 401

    def test_invalid_token_returns_401(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        req = self._make_request({"x-admin-token": "bad"})
        with pytest.raises(HTTPException) as exc:
            admin_auth.require_admin(req)
        assert exc.value.status_code == 401

    def test_valid_token_returns_username(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        req = self._make_request({"x-admin-token": token})
        assert admin_auth.require_admin(req) == "admin"

    def test_bearer_header_supported(self, monkeypatch):
        _set_admin(monkeypatch, admin=AdminConfig(username="admin", password="pw"))
        token = admin_auth.login("admin", "pw")
        req = self._make_request({"authorization": f"Bearer {token}"})
        assert admin_auth.require_admin(req) == "admin"

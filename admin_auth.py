"""
管理界面认证：独立 admin 密码 + 无状态签名 token。

设计要点：
- 密码支持明文（password）或 sha256 哈希（password_hash）两种形式，二者配其一。
  哈希形式复用 usage_stats._hash_key（sha256 hex）。
- 登录成功后颁发 HMAC 签名的自包含 token（payload 里带用户名与过期时间），
  默认 12h 过期。**不用服务端 session 表**：uvicorn --workers N 下每个 worker
  是独立进程、内存不共享，进程内的 session 表会让管理界面随机 401
  （登录落在 worker A、下一个请求落到 worker B 就失效）。
- 签名密钥由 admin 凭据派生，因此每个 worker 都能独立校验别人签发的 token；
  改密码等价于作废全部已签发会话。
- 未配置 [admin] 段时，所有受保护端点返回 503，避免无认证可写。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from typing import Optional

from fastapi import HTTPException, Request, status

from config import settings
from usage_stats import _hash_key

# token 默认有效期（秒）：12 小时
TOKEN_TTL_SECONDS = 12 * 60 * 60

_lock = threading.Lock()

# 登出撤销名单：token 的「解码后 payload+sig 的 sha256」（见 _token_key）-> 过期时刻。
#
# 无状态 token 没有服务端会话可删，只能另记一份名单。注意这份名单是**进程内**的：
# 多 worker 下登出只对本进程立即生效，其它 worker 仍会认这个 token 直到它自然
# 过期。客户端登出会清掉本地存储，正常使用不受影响；要做到全进程组即时失效，
# 得引入共享存储，对本项目（本地管理台）不划算。
_revoked: dict[str, float] = {}


# ---------------------------------------------------------------------------
#  密码校验
# ---------------------------------------------------------------------------

def _verify_password(raw_password: str) -> bool:
    """比对输入密码与配置的 admin 凭据。"""
    admin = settings.ADMIN
    if admin is None:
        return False

    # 优先校验 password_hash
    if admin.password_hash:
        return secrets.compare_digest(
            _hash_key(raw_password).lower(),
            admin.password_hash.lower(),
        )
    if admin.password:
        return secrets.compare_digest(raw_password, admin.password)
    return False


def is_admin_enabled() -> bool:
    """是否配置了 [admin] 段（决定管理界面是否可用）。"""
    return settings.ADMIN is not None


# ---------------------------------------------------------------------------
#  session token
# ---------------------------------------------------------------------------

def _now() -> float:
    """当前时刻（wall clock）。

    刻意不用 time.monotonic()：过期时间要写进 token 由**别的进程**校验，
    monotonic 的起点虽然同机一致，但语义上不可跨进程/跨重启比较。
    """
    return time.time()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    """base64url 解码，容忍被去掉的 padding。非法输入抛 ValueError。

    只接受**规范**编码：解码后再编回去必须与原串逐字符相同。base64 的最后一个
    字符带 2 个冗余位，同一个字节序列存在多种写法（"AA" 与 "AB" 解出来都是
    b'\\x00'），标准解码器不校验这些位。若放行，一个 token 就有了无数种等价写法，
    任何「按字符串」做的记录（比如撤销名单）都能被换个尾字符绕过 —— 见 _token_key。
    """
    raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    if _b64e(raw) != text:
        raise ValueError("非规范的 base64 编码")
    return raw


def _signing_key() -> bytes:
    """派生 token 签名密钥。

    必须来自共享的 config.toml 而不是进程内随机数 —— 否则 worker A 签发的
    token，worker B 校验不过。这里从 admin 凭据推导：改密码 = 作废所有会话。
    """
    admin = settings.ADMIN
    material = ""
    if admin is not None:
        material = admin.password_hash or admin.password or ""
    return hashlib.sha256(
        b"ai-proxy/admin-session/v1\x00" + material.encode("utf-8")
    ).digest()


def _token_key(payload: bytes, sig: bytes) -> str:
    """撤销名单的键：对**解码后**的 payload + sig 取摘要。

    不能拿原始 token 字符串做键。字符串有等价写法（见 _b64d 的规范性检查），
    拿字符串做键的话，只要改动尾字符就能同时在「验签通过」和「撤销名单查不到」
    两边成立，登出形同虚设。按解码后的字节做键，等价写法天然映射到同一个键。
    """
    return hashlib.sha256(
        len(payload).to_bytes(8, "big") + payload + sig
    ).hexdigest()


def _purge_expired(now: float) -> None:
    """清掉撤销名单里已经自然过期的条目（那些 token 本来就认不过了）。调用方需持锁。"""
    for digest in [d for d, exp in _revoked.items() if exp <= now]:
        _revoked.pop(digest, None)


def issue_token(username: str) -> str:
    """颁发一个签名 token：base64url(payload).base64url(hmac)。"""
    payload = json.dumps(
        {"u": username, "e": _now() + TOKEN_TTL_SECONDS},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    sig = hmac.new(_signing_key(), payload, hashlib.sha256).digest()
    return f"{_b64e(payload)}.{_b64e(sig)}"


def validate_token(token: Optional[str]) -> Optional[str]:
    """校验 token，返回 username；签名不对/格式坏/已过期/已登出都返回 None。"""
    if not token:
        return None

    payload_b64, sep, sig_b64 = token.partition(".")
    if not sep:
        return None
    try:
        payload = _b64d(payload_b64)
        sig = _b64d(sig_b64)
    except (ValueError, TypeError):
        return None

    # 先验签再解析：签名不过的 payload 不值得信任，也不该进 json.loads
    expected = hmac.new(_signing_key(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None

    try:
        data = json.loads(payload)
        username = data["u"]
        expire = float(data["e"])
    except (ValueError, TypeError, KeyError):
        return None

    now = _now()
    with _lock:
        _purge_expired(now)
        revoked = _token_key(payload, sig) in _revoked

    if revoked or expire <= now:
        return None
    if not isinstance(username, str) or not username:
        return None
    return username


def revoke_token(token: Optional[str]) -> None:
    """登出：把 token 记进本进程撤销名单（见 _revoked 的说明）。"""
    if not token:
        return
    payload_b64, sep, sig_b64 = token.partition(".")
    if not sep:
        return
    try:
        payload = _b64d(payload_b64)
        sig = _b64d(sig_b64)
    except (ValueError, TypeError):
        # 解不出来的 token 本来就验签不过，可以不记
        return

    # 记到该 token 自己的过期时刻即可，之后留条目也没意义
    try:
        expire = float(json.loads(payload)["e"])
    except (ValueError, TypeError, KeyError):
        expire = _now() + TOKEN_TTL_SECONDS

    with _lock:
        _purge_expired(_now())
        _revoked[_token_key(payload, sig)] = expire


# ---------------------------------------------------------------------------
#  FastAPI 依赖
# ---------------------------------------------------------------------------

def _extract_token(request: Request) -> Optional[str]:
    """从请求头提取 token，支持 X-Admin-Token 与 Authorization: Bearer。"""
    direct = request.headers.get("x-admin-token", "").strip()
    if direct:
        return direct
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_admin(request: Request) -> str:
    """受保护端点的依赖：校验 token，返回 username。

    - 未配置 [admin] → 503（管理界面未启用）
    - token 缺失 → 401
    - token 无效/过期 → 401
    """
    if not is_admin_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理界面未配置 [admin] 段，请在 config.toml 中配置后热重载。",
        )

    token = _extract_token(request)
    username = validate_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未认证或会话已过期，请重新登录。",
        )
    return username


def login(username: str, password: str) -> Optional[str]:
    """校验用户名密码，成功返回 token，失败返回 None。"""
    admin = settings.ADMIN
    if admin is None:
        return None
    if not secrets.compare_digest(username, admin.username):
        return None
    if not _verify_password(password):
        return None
    return issue_token(username)

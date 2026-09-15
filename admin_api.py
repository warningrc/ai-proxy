"""
管理界面 API 路由。

端点：
  GET  /admin                → 单页 HTML 前端
  POST /admin/api/login      → 颁发 token
  POST /admin/api/logout     → 撤销 token
  GET  /admin/api/config     → 当前配置 JSON（api_key 脱敏）
  PUT  /admin/api/config     → 全量配置 → 校验 + 写回 + 热重载
  GET  /admin/api/stats      → 用量统计（透传 usage_stats.query_stats）
  GET  /admin/api/stats/daily→ 时间 × 模型用量序列（透传 usage_stats.query_daily_usage）

脱敏约定：GET 返回的 api_key 为掩码（_mask_key）。PUT 时若回传值仍是掩码
（含 '***'），视为"未改动"，用当前 settings 中的真实值替换；否则视为用户新输入
的明文，原样采用。这样明文 key 永不经过界面回显。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import usage_stats as usage_stats_module
from admin_auth import is_admin_enabled, login, require_admin, revoke_token
from admin_reloader import apply_config_and_runtime
from config import settings
from usage_stats import _mask_key

logger = logging.getLogger("ai-proxy.admin")

router = APIRouter()


def _app_main() -> Any:
    """返回应用真正运行的 main 模块。

    兼容两种启动方式，避免「python main.py」时 `import main` 产生第二个模块实例
    （导致 http_clients / usage_stats / 模型发现缓存 与运行中的实例不一致）：
    - `uvicorn main:app` → sys.modules["main"]
    - `python main.py`     → sys.modules["__main__"]
    """
    mod = sys.modules.get("main")
    if mod is not None:
        return mod
    mod = sys.modules.get("__main__")
    if mod is not None and hasattr(mod, "discover_upstream_models"):
        return mod
    raise RuntimeError("无法定位运行中的 main 模块")

_WEB_DIR = Path(__file__).resolve().parent / "web"


# ---------------------------------------------------------------------------
#  Pydantic 请求模型
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
#  脱敏 / 还原辅助
# ---------------------------------------------------------------------------

def _is_masked(value: Optional[str]) -> bool:
    """判断一个 api_key 字符串是否为掩码值（未改动）。"""
    return bool(value) and "***" in value


def _resolve_masked_keys(incoming: Dict[str, Any]) -> Dict[str, Any]:
    """把回传配置中的掩码 api_key 还原为当前 settings 中的真实值。

    处理位置：providers 顶层共享 api_key、providers[].openai.api_key、
    providers[].anthropic.api_key、tenants[].api_key、admin.password、
    admin.password_hash。

    这里必须和 _mask_config 成对：只脱敏不还原的话，界面「不改就保存」会把掩码
    本身当新值写进 config.toml，凭据当场报废。
    """
    # 建立 provider_id -> 真实 key 映射（按当前 settings）
    current = settings.to_dict()

    # providers
    cur_providers: Dict[str, Dict[str, Any]] = {
        p["id"]: p for p in current.get("providers", [])
    }
    for p in incoming.get("providers", []):
        pid = p.get("id")
        cur = cur_providers.get(pid, {})

        if _is_masked(p.get("api_key")):
            if "api_key" in cur:
                p["api_key"] = cur["api_key"]
            else:
                p.pop("api_key", None)

        for proto in ("openai", "anthropic"):
            sub = p.get(proto)
            if isinstance(sub, dict) and _is_masked(sub.get("api_key")):
                cur_sub = cur.get(proto, {})
                if "api_key" in cur_sub:
                    sub["api_key"] = cur_sub["api_key"]
                else:
                    sub.pop("api_key", None)

    # tenants
    cur_tenants: Dict[str, Dict[str, Any]] = {
        t["id"]: t for t in current.get("tenants", [])
    }
    for t in incoming.get("tenants", []):
        tid = t.get("id")
        cur_t = cur_tenants.get(tid, {})
        if _is_masked(t.get("api_key")) and "api_key" in cur_t:
            t["api_key"] = cur_t["api_key"]

    # admin
    cur_admin = current.get("admin", {})
    admin = incoming.get("admin")
    if isinstance(admin, dict):
        for field in ("password", "password_hash"):
            if not _is_masked(admin.get(field)):
                continue
            if field in cur_admin:
                admin[field] = cur_admin[field]
            else:
                admin.pop(field, None)

    return incoming


def _mask_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """返回一份 api_key 全部脱敏的配置副本（用于 GET /admin/api/config）。

    admin.password_hash 也必须脱敏：它不是「不可逆的摘要」，而是 admin 会话
    token 的签名密钥原料（见 admin_auth._signing_key）。拿到它就能离线伪造任意
    管理会话 token，等价于拿到管理员权限 —— 比泄漏明文密码还省事（不用爆破）。
    """
    out = _deep_copy(config_dict)

    for p in out.get("providers", []):
        if "api_key" in p:
            p["api_key"] = _mask_key(p["api_key"])
        for proto in ("openai", "anthropic"):
            sub = p.get(proto)
            if isinstance(sub, dict) and "api_key" in sub:
                sub["api_key"] = _mask_key(sub["api_key"])

    for t in out.get("tenants", []):
        if "api_key" in t:
            t["api_key"] = _mask_key(t["api_key"])

    admin = out.get("admin")
    if isinstance(admin, dict):
        for field in ("password", "password_hash"):
            if field in admin:
                admin[field] = _mask_key(admin[field])

    return out


def _deep_copy(obj: Any) -> Any:
    """简单深拷贝（配置仅含 JSON 兼容类型）。"""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
#  路由
# ---------------------------------------------------------------------------

@router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """管理界面单页 HTML。"""
    index_path = _WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html 不存在")
    return HTMLResponse(
        index_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@router.post("/admin/api/login")
async def admin_login(req: LoginRequest):
    if not is_admin_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理界面未配置 [admin] 段。",
        )
    token = login(req.username, req.password)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    logger.info("admin login | user=%r", req.username)
    return {"token": token, "username": req.username}


@router.post("/admin/api/logout")
async def admin_logout(request: Request, username: str = Depends(require_admin)):
    """登出：从任意支持的请求头提取 token 并撤销。"""
    token = request.headers.get("x-admin-token", "").strip()
    if not token:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    revoke_token(token)
    logger.info("admin logout | user=%r", username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/admin/api/config")
async def get_admin_config(username: str = Depends(require_admin)):
    """返回当前配置（api_key 脱敏）。"""
    return _mask_config(settings.to_dict())


@router.put("/admin/api/config")
async def put_admin_config(
    body: Dict[str, Any],
    username: str = Depends(require_admin),
):
    """接收全量配置，还原掩码后校验 + 写回 + 热重载。"""
    resolved = _resolve_masked_keys(body)

    try:
        result = await apply_config_and_runtime(resolved)
    except Exception as e:
        logger.error("admin reload failed | user=%r | %s", username, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"热重载异常: {type(e).__name__}: {e}",
        )

    if not result.success:
        # 校验失败 / 写盘失败 → 422，旧配置不受影响
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": result.error, "detail": result.detail},
        )

    logger.info("admin config reloaded | user=%r", username)
    return {"ok": True, "message": "配置已保存并热重载"}


@router.get("/admin/api/stats")
async def admin_stats(
    group_by: str = "model",
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    tenant: Optional[str] = None,
    provider: Optional[str] = None,
    username: str = Depends(require_admin),
):
    """用量统计查询（复用 usage_stats.query_stats）。"""
    usage_stats = _app_main().usage_stats

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: usage_stats.query_stats(
                group_by=group_by,
                since=since,
                until=until,
                model=model,
                tenant=tenant,
                provider=provider,
            ),
        )
        return JSONResponse(content=res)
    except Exception as e:
        logger.error("admin stats query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"统计查询失败: {e}",
        )


@router.get("/admin/api/stats/daily")
async def admin_stats_daily(
    metric: str = "total_tokens",
    bucket: str = "day",
    since: Optional[str] = None,
    until: Optional[str] = None,
    model: Optional[str] = None,
    tenant: Optional[str] = None,
    provider: Optional[str] = None,
    username: str = Depends(require_admin),
):
    """按时间 × 模型的用量序列（供图表使用）。

    与 /admin/api/stats 的区别：这里对未知的 metric / bucket **报错**而不是静默回退。
    新接口没有历史包袱，返回 400 远好过返回一份看起来正常、实际答非所问的数据。
    """
    usage_stats = _app_main().usage_stats

    if metric not in usage_stats_module._METRIC_SQL:
        raise HTTPException(
            status_code=400,
            detail=f"未知 metric {metric!r}，可选: {sorted(usage_stats_module._METRIC_SQL)}",
        )
    if bucket not in usage_stats_module._BUCKET_SQL:
        raise HTTPException(
            status_code=400,
            detail=f"未知 bucket {bucket!r}，可选: {sorted(usage_stats_module._BUCKET_SQL)}",
        )

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: usage_stats.query_daily_usage(
                metric=metric,
                bucket=bucket,
                since=since,
                until=until,
                model=model,
                tenant=tenant,
                provider=provider,
            ),
        )
        return JSONResponse(content=res)
    except Exception as e:
        logger.error("admin daily stats query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"统计查询失败: {e}",
        )


@router.get("/admin/api/provider-models")
async def admin_provider_models(
    provider: str,
    username: str = Depends(require_admin),
):
    """返回指定 provider 的上游可用模型名列表（供管理界面模型路由下拉选择）。"""
    provider = (provider or "").strip()
    if not provider:
        raise HTTPException(status_code=422, detail="缺少 provider 参数")
    if provider not in settings.PROVIDERS:
        raise HTTPException(
            status_code=404,
            detail=f"provider {provider!r} 不存在（可能尚未保存，请先保存配置）",
        )

    try:
        discovered = await _app_main().discover_upstream_models()
    except Exception as e:
        logger.error("provider-models | discovery failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取上游模型失败: {type(e).__name__}: {e}",
        )

    models = sorted({
        entry["upstream_model"]
        for entry in discovered.values()
        if entry.get("provider") == provider and entry.get("upstream_model")
    })
    return {"provider": provider, "models": models}

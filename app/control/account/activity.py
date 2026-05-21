"""Lightweight self-serve activity telemetry for admin surfaces.

Tracks media capability counters and the most recent media error in ``ext`` so
the admin UI can distinguish conversation traffic from image/video traffic
without changing the core quota model.
"""

from app.platform.runtime.clock import now_ms
from app.platform.errors import UpstreamError

from .commands import AccountPatch
from .runtime import get_refresh_service


def _status_code(exc: BaseException | None) -> int | None:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    return None


def _error_message(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    if isinstance(exc, UpstreamError):
        msg = str(exc)
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            body = str(details.get("body", "") or "").replace("\n", " ").strip()
            if body:
                return f"{msg} | {body[:240]}"
        return msg
    return str(exc)


async def record_media_activity(
    token: str,
    *,
    capability: str,
    route: str,
    model: str,
    success: bool,
    exc: BaseException | None = None,
) -> None:
    """Persist lightweight media counters and the latest media error.

    Stored in ``AccountRecord.ext`` only so this remains additive and safe for
    self-serve observability without altering quota semantics.
    """
    if capability not in {"image", "video"}:
        return

    svc = get_refresh_service()
    repo = getattr(svc, "_repo", None) if svc is not None else None
    if repo is None:
        return

    records = await repo.get_accounts([token])
    record = records[0] if records else None
    if record is None or record.is_deleted():
        return

    ext = record.ext if isinstance(record.ext, dict) else {}
    now = now_ms()
    ok_key = f"{capability}_ok"
    fail_key = f"{capability}_fail"
    patch_ext = {
        ok_key: int(ext.get(ok_key, 0) or 0) + (1 if success else 0),
        fail_key: int(ext.get(fail_key, 0) or 0) + (0 if success else 1),
        "last_media_at": now,
        "last_media_route": route,
        "last_media_model": model,
        "last_media_capability": capability,
    }
    if not success:
        patch_ext.update(
            {
                "last_media_error_at": now,
                "last_media_error_route": route,
                "last_media_error_model": model,
                "last_media_error_capability": capability,
                "last_media_error_status": _status_code(exc),
                "last_media_error_message": _error_message(exc)[:280],
            }
        )

    await repo.patch_accounts([AccountPatch(token=token, ext_merge=patch_ext)])


__all__ = ["record_media_activity"]

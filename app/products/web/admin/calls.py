"""Structured call-history admin API."""

from __future__ import annotations

from typing import Any

import orjson
from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.products.openai.call_history import display_text_from_request, display_text_from_response
from app.platform.storage import call_history_store, should_expose_sensitive, summarize_call_history

router = APIRouter(prefix="/calls", tags=["Admin - Calls"])


def _list_item(record, *, include_sensitive: bool) -> dict[str, Any]:
    data = summarize_call_history(record, include_sensitive=include_sensitive)
    data.pop("request_body", None)
    data.pop("response_body", None)
    return data


def _display_text_from_request(value: Any) -> str:
    return display_text_from_request(value)


def _display_text_from_response(value: Any) -> str:
    return display_text_from_response(value)


@router.get("")
async def list_calls(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    model: str | None = Query(default=None),
    route: str | None = Query(default=None),
    status: str | None = Query(default=None),
    stream: bool | None = Query(default=None),
    q: str | None = Query(default=None),
    from_ms: int | None = Query(default=None, alias="from"),
    to_ms: int | None = Query(default=None, alias="to"),
):
    page_data = await call_history_store.list_calls(
        page=page,
        page_size=page_size,
        model=model,
        route=route,
        status=status,
        stream=stream,
        q=q,
        from_ms=from_ms,
        to_ms=to_ms,
    )
    include_sensitive = should_expose_sensitive()
    items = [_list_item(item, include_sensitive=include_sensitive) for item in page_data.items]
    return Response(
        content=orjson.dumps(
            {
                "items": items,
                "total": page_data.total,
                "page": page_data.page,
                "page_size": page_data.page_size,
                "total_pages": page_data.total_pages,
                "show_sensitive": include_sensitive,
            }
        ),
        media_type="application/json",
    )


@router.get("/{call_id}")
async def get_call(call_id: str):
    include_sensitive = should_expose_sensitive()
    record = await call_history_store.get(call_id)
    if record is None:
        return Response(
            content=orjson.dumps({"error": {"message": "Call not found", "type": "invalid_request_error"}}),
            status_code=404,
            media_type="application/json",
        )
    data = summarize_call_history(record, include_sensitive=include_sensitive)
    if include_sensitive:
        data["request_text"] = _display_text_from_request(data.get("request_body") or "")
        data["response_text"] = _display_text_from_response(data.get("response_body") or "")
    return Response(content=orjson.dumps(data), media_type="application/json")


__all__ = ["router"]

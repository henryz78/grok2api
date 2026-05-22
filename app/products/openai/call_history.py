"""Call-history instrumentation helpers for OpenAI-compatible routes."""

from __future__ import annotations

from typing import Any, AsyncGenerator, AsyncIterable

import orjson
from fastapi import Request

from app.platform.logging.logger import logger
from app.platform.runtime.clock import now_ms
from app.platform.storage import call_history_store


def request_client_ip(request: Request | None) -> str:
    if request is None:
        return ""
    forwarded = request.headers.get("x-forwarded-for") if request.headers else ""
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    real_ip = request.headers.get("x-real-ip") if request.headers else ""
    if real_ip:
        return real_ip.strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or ""


def _truncate_history_text(value: str, limit: int = 240) -> str:
    if not value:
        return ""
    text = value.replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[: max(0, limit - 3)]}..."


def _extract_nested_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_extract_nested_text(item) for item in value) if part)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        if isinstance(value.get("delta"), dict):
            return _extract_nested_text(value["delta"])
        if isinstance(value.get("message"), dict):
            return _extract_nested_text(value["message"])
        if isinstance(value.get("output_text"), str):
            return value["output_text"]
        if isinstance(value.get("content"), list):
            return _extract_nested_text(value["content"])
        if isinstance(value.get("output"), list):
            return _extract_nested_text(value["output"])
    return ""


def usage_from_payload(payload: dict) -> dict[str, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    if "input_tokens" in usage or "output_tokens" in usage:
        reasoning = usage.get("output_tokens_details", {})
        return {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "reasoning_tokens": (
                reasoning.get("reasoning_tokens") if isinstance(reasoning, dict) else None
            ),
            "total_tokens": usage.get("total_tokens"),
        }

    completion_details = usage.get("completion_tokens_details", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        ),
        "total_tokens": usage.get("total_tokens"),
    }


def response_preview_from_payload(payload: dict) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        text = _extract_nested_text(first)
        if text:
            return _truncate_history_text(text)
        finish = first.get("finish_reason")
        if finish:
            return f"finish_reason={finish}"
    output = payload.get("output")
    if isinstance(output, list):
        text = _extract_nested_text(output)
        if text:
            return _truncate_history_text(text)
    return _truncate_history_text(str(payload.get("id") or payload.get("object") or ""))


def app_error_status(exc: BaseException) -> int:
    status = getattr(exc, "status", None)
    return int(status) if isinstance(status, int) else 500


def app_error_type(exc: BaseException) -> str:
    kind = getattr(exc, "kind", None)
    return str(kind or exc.__class__.__name__)


async def record_call_history(
    *,
    started_at_ms: int,
    route: str,
    model: str,
    stream: bool,
    request_body: Any,
    client_ip: str,
    success: bool,
    status_code: int | None = None,
    error_type: str = "",
    error_message: str = "",
    response_body: Any = None,
    response_preview: str = "",
    usage: dict[str, int | None] | None = None,
    meta: dict | None = None,
) -> None:
    try:
        finished_at_ms = now_ms()
        tokens = usage or {}
        entry = call_history_store.build_entry(
            created_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            route=route,
            model=model,
            stream=stream,
            success=success,
            status_code=status_code,
            error_type=error_type,
            error_message=error_message,
            prompt_tokens=tokens.get("prompt_tokens"),
            completion_tokens=tokens.get("completion_tokens"),
            reasoning_tokens=tokens.get("reasoning_tokens"),
            total_tokens=tokens.get("total_tokens"),
            client_ip=client_ip,
            request_body=request_body,
            response_body=response_body or "",
            response_preview=response_preview,
            meta=meta or {},
        )
        await call_history_store.record(entry)
    except Exception as exc:
        logger.warning("call history record failed: route={} model={} error={}", route, model, exc)


def _capture_sse_data(
    chunk: str,
    *,
    text_parts: list[str],
    response_parts: list[str],
    usage_holder: dict[str, int | None],
) -> tuple[bool, str, str]:
    if sum(len(item) for item in response_parts) < 1_048_576:
        response_parts.append(chunk)

    failed = False
    error_type = ""
    error_message = ""
    event_name = ""
    for raw_line in chunk.splitlines():
        line = raw_line.strip()
        if line.startswith("event:"):
            event_name = line[6:].strip()
            if event_name == "error":
                failed = True
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = orjson.loads(data)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if event_name == "error" or payload.get("type") == "error" or "error" in payload:
            failed = True
            err = payload.get("error") if isinstance(payload.get("error"), dict) else payload
            error_message = str(err.get("message") or error_message or "stream error")
            error_type = str(err.get("type") or error_type or "upstream_error")
        extracted_usage = usage_from_payload(payload)
        if extracted_usage.get("total_tokens") is not None:
            usage_holder.update(extracted_usage)
        text = _extract_nested_text(payload)
        if text:
            text_parts.append(text)
    return failed, error_type, error_message


async def recording_sse(
    stream: AsyncIterable[str],
    *,
    started_at_ms: int,
    route: str,
    model: str,
    request_body: Any,
    client_ip: str,
    meta: dict | None = None,
) -> AsyncGenerator[str, None]:
    text_parts: list[str] = []
    response_parts: list[str] = []
    usage: dict[str, int | None] = {}
    success = True
    status_code = 200
    error_type = ""
    error_message = ""
    try:
        async for chunk in stream:
            failed, etype, emsg = _capture_sse_data(
                chunk,
                text_parts=text_parts,
                response_parts=response_parts,
                usage_holder=usage,
            )
            if failed:
                success = False
                status_code = 502
                error_type = etype or error_type
                error_message = emsg or error_message
            yield chunk
    except BaseException as exc:
        success = False
        status_code = app_error_status(exc)
        error_type = app_error_type(exc)
        error_message = str(exc)
        raise
    finally:
        await record_call_history(
            started_at_ms=started_at_ms,
            route=route,
            model=model,
            stream=True,
            request_body=request_body,
            client_ip=client_ip,
            success=success,
            status_code=status_code,
            error_type=error_type,
            error_message=error_message,
            response_body="".join(response_parts),
            response_preview=_truncate_history_text("".join(text_parts)),
            usage=usage or None,
            meta=meta,
        )


__all__ = [
    "app_error_status",
    "app_error_type",
    "record_call_history",
    "recording_sse",
    "request_client_ip",
    "response_preview_from_payload",
    "usage_from_payload",
]

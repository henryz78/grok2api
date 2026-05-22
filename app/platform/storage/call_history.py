"""Append-only SQLite store for structured model call history."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import uuid
from decimal import Decimal, ROUND_HALF_UP
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.platform.config.snapshot import get_config

from .call_history_paths import call_history_db_path

_TABLE = "call_history"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | bytes | None, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    text = str(text)
    return text if len(text) <= limit else f"{text[: max(0, limit - 3)]}..."


def _fmt_tokens(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def _fmt_duration(duration_ms: int | None) -> str:
    if not duration_ms or duration_ms < 0:
        return "—"
    seconds = Decimal(duration_ms) / Decimal(1000)
    if seconds >= 10:
        return f"{seconds.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}s"
    return f"{seconds.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}s"


def _fmt_tokens_per_s(duration_ms: int | None, total_tokens: int | None) -> str:
    if not duration_ms or duration_ms <= 0 or total_tokens is None:
        return "—"
    rate = int(total_tokens) / (duration_ms / 1000)
    return f"{rate:.0f} t/s"


@dataclass(slots=True)
class CallHistoryRecord:
    id: str
    created_at_ms: int
    finished_at_ms: int
    duration_ms: int
    route: str
    model: str
    stream: bool
    success: bool
    status_code: int | None
    error_type: str
    error_message: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    client_ip: str
    request_body: str
    response_body: str
    request_preview: str
    response_preview: str
    request_size_bytes: int
    response_size_bytes: int
    meta: dict[str, Any]

    @property
    def route_label(self) -> str:
        return self.route or "—"

    @property
    def stream_label(self) -> str:
        return "流" if self.stream else "非流"

    @property
    def status_label(self) -> str:
        return "成功" if self.success else "失败"

    @property
    def token_summary(self) -> str:
        parts = [
            _fmt_duration(self.duration_ms),
            self.stream_label,
            _fmt_tokens_per_s(self.duration_ms, self.total_tokens),
            f"{_fmt_tokens(self.prompt_tokens)} / {_fmt_tokens(self.completion_tokens)}",
        ]
        return " · ".join(parts)

    def summary_dict(self, *, include_sensitive: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "created_at_ms": self.created_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "duration_ms": self.duration_ms,
            "route": self.route,
            "model": self.model,
            "stream": self.stream,
            "success": self.success,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "request_preview": self.request_preview,
            "response_preview": self.response_preview,
            "request_size_bytes": self.request_size_bytes,
            "response_size_bytes": self.response_size_bytes,
            "meta": self.meta,
            "client_ip": self.client_ip if include_sensitive else "",
            "token_summary": self.token_summary,
        }
        if include_sensitive:
            data["request_body"] = self.request_body
            data["response_body"] = self.response_body
        return data


@dataclass(slots=True)
class CallHistoryPage:
    items: list[CallHistoryRecord]
    total: int
    page: int
    page_size: int
    total_pages: int


class CallHistoryStore:
    """SQLite-backed append-only store for structured call history."""

    def __init__(self, db_path: Path | None = None, *, config_provider: Callable[[], Any] = get_config) -> None:
        self._path = Path(db_path) if db_path is not None else call_history_db_path()
        self._config_provider = config_provider
        self._lock = asyncio.Lock()
        self._init_lock = threading.Lock()
        self._initialized_dbs: set[Path] = set()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._path in self._initialized_dbs:
            return
        with self._init_lock:
            if self._path in self._initialized_dbs:
                return
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    id                TEXT PRIMARY KEY,
                    created_at_ms     INTEGER NOT NULL,
                    finished_at_ms    INTEGER NOT NULL,
                    duration_ms       INTEGER NOT NULL,
                    route             TEXT NOT NULL,
                    model             TEXT NOT NULL,
                    stream            INTEGER NOT NULL,
                    success           INTEGER NOT NULL,
                    status_code       INTEGER,
                    error_type        TEXT NOT NULL DEFAULT '',
                    error_message     TEXT NOT NULL DEFAULT '',
                    prompt_tokens     INTEGER,
                    completion_tokens INTEGER,
                    reasoning_tokens  INTEGER,
                    total_tokens      INTEGER,
                    client_ip         TEXT NOT NULL DEFAULT '',
                    request_body      TEXT NOT NULL DEFAULT '',
                    response_body     TEXT NOT NULL DEFAULT '',
                    request_preview   TEXT NOT NULL DEFAULT '',
                    response_preview  TEXT NOT NULL DEFAULT '',
                    request_size_bytes INTEGER NOT NULL DEFAULT 0,
                    response_size_bytes INTEGER NOT NULL DEFAULT 0,
                    meta              TEXT NOT NULL DEFAULT '{{}}'
                );
                CREATE INDEX IF NOT EXISTS idx_call_history_created
                    ON {_TABLE} (created_at_ms DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_call_history_model
                    ON {_TABLE} (model, created_at_ms DESC);
                CREATE INDEX IF NOT EXISTS idx_call_history_route
                    ON {_TABLE} (route, created_at_ms DESC);
                CREATE INDEX IF NOT EXISTS idx_call_history_success
                    ON {_TABLE} (success, created_at_ms DESC);
                """
            )
            conn.commit()
            self._initialized_dbs.add(self._path)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CallHistoryRecord:
        return CallHistoryRecord(
            id=str(row["id"]),
            created_at_ms=int(row["created_at_ms"]),
            finished_at_ms=int(row["finished_at_ms"]),
            duration_ms=int(row["duration_ms"]),
            route=str(row["route"] or ""),
            model=str(row["model"] or ""),
            stream=bool(row["stream"]),
            success=bool(row["success"]),
            status_code=int(row["status_code"]) if row["status_code"] is not None else None,
            error_type=str(row["error_type"] or ""),
            error_message=str(row["error_message"] or ""),
            prompt_tokens=int(row["prompt_tokens"]) if row["prompt_tokens"] is not None else None,
            completion_tokens=int(row["completion_tokens"]) if row["completion_tokens"] is not None else None,
            reasoning_tokens=int(row["reasoning_tokens"]) if row["reasoning_tokens"] is not None else None,
            total_tokens=int(row["total_tokens"]) if row["total_tokens"] is not None else None,
            client_ip=str(row["client_ip"] or ""),
            request_body=str(row["request_body"] or ""),
            response_body=str(row["response_body"] or ""),
            request_preview=str(row["request_preview"] or ""),
            response_preview=str(row["response_preview"] or ""),
            request_size_bytes=int(row["request_size_bytes"] or 0),
            response_size_bytes=int(row["response_size_bytes"] or 0),
            meta=_json_loads(row["meta"], {}),
        )

    @staticmethod
    def _normalize_request_preview(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return _truncate(value.replace("\n", " ").strip(), 180)
        try:
            return _truncate(_json_dumps(value), 180)
        except Exception:
            return _truncate(str(value), 180)

    @staticmethod
    def _normalize_body(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return _json_dumps(value)
        except Exception:
            return str(value)

    def build_entry(
        self,
        *,
        created_at_ms: int,
        finished_at_ms: int,
        route: str,
        model: str,
        stream: bool,
        success: bool,
        status_code: int | None = None,
        error_type: str = "",
        error_message: str = "",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        total_tokens: int | None = None,
        client_ip: str = "",
        request_body: Any = "",
        response_body: Any = "",
        request_preview: str = "",
        response_preview: str = "",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        req_body = self._normalize_body(request_body)
        resp_body = self._normalize_body(response_body)
        req_preview = request_preview or self._normalize_request_preview(request_body)
        resp_preview = response_preview or self._normalize_request_preview(response_body)
        if not req_preview and req_body:
            req_preview = _truncate(req_body.replace("\n", " "), 180)
        if not resp_preview and resp_body:
            resp_preview = _truncate(resp_body.replace("\n", " "), 180)
        return {
            "id": f"call_{created_at_ms}_{finished_at_ms}_{uuid.uuid4().hex[:12]}",
            "created_at_ms": int(created_at_ms),
            "finished_at_ms": int(finished_at_ms),
            "duration_ms": max(0, int(finished_at_ms) - int(created_at_ms)),
            "route": route,
            "model": model,
            "stream": 1 if stream else 0,
            "success": 1 if success else 0,
            "status_code": status_code,
            "error_type": error_type or "",
            "error_message": error_message or "",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "client_ip": client_ip or "",
            "request_body": req_body,
            "response_body": resp_body,
            "request_preview": req_preview,
            "response_preview": resp_preview,
            "request_size_bytes": len(req_body.encode("utf-8")) if req_body else 0,
            "response_size_bytes": len(resp_body.encode("utf-8")) if resp_body else 0,
            "meta": _json_dumps(meta or {}),
        }

    async def initialize(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with closing(self._connect()) as conn:
            self._ensure_schema(conn)

    async def record(self, entry: dict[str, Any]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._record_sync, entry)

    def _record_sync(self, entry: dict[str, Any]) -> None:
        with closing(self._connect()) as conn:
            self._ensure_schema(conn)
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {_TABLE} (
                    id,
                    created_at_ms,
                    finished_at_ms,
                    duration_ms,
                    route,
                    model,
                    stream,
                    success,
                    status_code,
                    error_type,
                    error_message,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    total_tokens,
                    client_ip,
                    request_body,
                    response_body,
                    request_preview,
                    response_preview,
                    request_size_bytes,
                    response_size_bytes,
                    meta
                ) VALUES (
                    :id,
                    :created_at_ms,
                    :finished_at_ms,
                    :duration_ms,
                    :route,
                    :model,
                    :stream,
                    :success,
                    :status_code,
                    :error_type,
                    :error_message,
                    :prompt_tokens,
                    :completion_tokens,
                    :reasoning_tokens,
                    :total_tokens,
                    :client_ip,
                    :request_body,
                    :response_body,
                    :request_preview,
                    :response_preview,
                    :request_size_bytes,
                    :response_size_bytes,
                    :meta
                )
                """,
                entry,
            )
            conn.commit()

    async def get(self, call_id: str) -> CallHistoryRecord | None:
        def _sync() -> CallHistoryRecord | None:
            with closing(self._connect()) as conn:
                self._ensure_schema(conn)
                row = conn.execute(f"SELECT * FROM {_TABLE} WHERE id = ?", (call_id,)).fetchone()
                return self._row_to_record(row) if row else None

        return await asyncio.to_thread(_sync)

    async def list_calls(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        model: str | None = None,
        route: str | None = None,
        status: str | None = None,
        stream: bool | None = None,
        q: str | None = None,
        from_ms: int | None = None,
        to_ms: int | None = None,
    ) -> CallHistoryPage:
        def _sync() -> CallHistoryPage:
            with closing(self._connect()) as conn:
                self._ensure_schema(conn)
                where: list[str] = []
                params: list[Any] = []
                if model:
                    where.append("model = ?")
                    params.append(model)
                if route:
                    where.append("route = ?")
                    params.append(route)
                if status == "success":
                    where.append("success = 1")
                elif status == "failed":
                    where.append("success = 0")
                if stream is not None:
                    where.append("stream = ?")
                    params.append(1 if stream else 0)
                if from_ms is not None:
                    where.append("created_at_ms >= ?")
                    params.append(from_ms)
                if to_ms is not None:
                    where.append("created_at_ms <= ?")
                    params.append(to_ms)
                if q:
                    needle = f"%{q.strip()}%"
                    where.append(
                        "("
                        "model LIKE ? OR route LIKE ? OR error_message LIKE ? OR request_preview LIKE ? OR response_preview LIKE ?"
                        ")"
                    )
                    params.extend([needle] * 5)

                where_sql = f"WHERE {' AND '.join(where)}" if where else ""
                total = conn.execute(f"SELECT COUNT(*) FROM {_TABLE} {where_sql}", params).fetchone()[0]
                page_num = max(1, int(page))
                page_limit = max(1, min(int(page_size), 200))
                offset = (page_num - 1) * page_limit
                rows = conn.execute(
                    f"SELECT * FROM {_TABLE} {where_sql} ORDER BY created_at_ms DESC, id DESC LIMIT ? OFFSET ?",
                    params + [page_limit, offset],
                ).fetchall()
                items = [self._row_to_record(row) for row in rows]
                total_pages = max(1, (int(total) + page_limit - 1) // page_limit)
                return CallHistoryPage(items=items, total=int(total), page=page_num, page_size=page_limit, total_pages=total_pages)

        return await asyncio.to_thread(_sync)

    async def close(self) -> None:
        return None


call_history_store = CallHistoryStore()


def summarize_call_history(record: CallHistoryRecord, *, include_sensitive: bool = False) -> dict[str, Any]:
    data = record.summary_dict(include_sensitive=include_sensitive)
    if not include_sensitive:
        data.pop("request_body", None)
        data.pop("response_body", None)
    return data


def should_expose_sensitive() -> bool:
    cfg = get_config()
    return cfg.get_bool("features.call_history_show_sensitive", False)


__all__ = [
    "CallHistoryPage",
    "CallHistoryRecord",
    "CallHistoryStore",
    "call_history_store",
    "should_expose_sensitive",
    "summarize_call_history",
]

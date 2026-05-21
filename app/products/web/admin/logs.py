"""File-based log browsing for the self-serve admin console."""

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from app.platform.paths import log_dir

router = APIRouter(prefix="/logs", tags=["Admin - Logs"])

_LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+"
    r"(?P<level>[A-Z]+)\s+\|\s+"
    r"(?P<source>.+?)\s+-\s+"
    r"(?P<message>.*)$"
)


def _list_log_files() -> list[Path]:
    root = log_dir()
    if not root.exists():
        return []
    return sorted(
        (path for path in root.glob("*.log") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _parse_line(file_name: str, raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = _LOG_LINE_RE.match(text)
    if match:
        ts = match.group("timestamp")
        level = match.group("level")
        source = match.group("source")
        message = match.group("message")
    else:
        ts, level, source, message = "", "", "", text
    return {
        "file": file_name,
        "timestamp": ts.strip(),
        "level": level.strip(),
        "source": source.strip(),
        "message": message.strip(),
        "raw": text,
    }


@router.get("")
async def list_logs(
    file: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    level: str | None = Query(default=None),
    q: str | None = Query(default=None),
):
    files = _list_log_files()
    file_names = [path.name for path in files]
    selected = [path for path in files if file is None or path.name == file]
    query = (q or "").strip().lower()
    level_filter = (level or "").strip().upper()

    items: list[dict[str, Any]] = []
    for path in selected:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for raw in reversed(lines):
            if not raw.strip():
                continue
            item = _parse_line(path.name, raw)
            if level_filter and item["level"].upper() != level_filter:
                continue
            haystack = f'{item["raw"]} {item["message"]} {item["source"]}'.lower()
            if query and query not in haystack:
                continue
            items.append(item)
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    return {
        "files": file_names,
        "selected_file": file,
        "items": items,
    }


__all__ = ["router"]

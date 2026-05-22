"""Shared paths for call-history storage."""

from pathlib import Path

from app.platform.paths import data_path


def call_history_db_path() -> Path:
    """Return the SQLite database path for call history."""
    path = data_path("call_history.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["call_history_db_path"]

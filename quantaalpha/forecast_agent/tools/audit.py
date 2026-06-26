"""Audit logging for forecast flow tool execution."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

FLOW_AUDIT_FILENAME = "flow_audit.jsonl"


def _now_iso() -> str:
    return datetime.now().isoformat()


def audit_path(out_root: str | Path) -> Path:
    return Path(out_root) / FLOW_AUDIT_FILENAME


def _safe_summary(payload: Any, max_len: int = 500) -> Any:
    if payload is None:
        return None
    text = str(payload)
    if len(text) <= max_len:
        return payload
    return text[: max_len - 3] + "..."


def write_audit_event(
    out_root: str | Path,
    *,
    task_id: str,
    stage: str,
    tool: str,
    status: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_ms: int | None = None,
    input_summary: Any = None,
    output_summary: Any = None,
    error: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = audit_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    event: dict[str, Any] = {
        "timestamp": _now_iso(),
        "taskId": task_id,
        "stage": str(stage or ""),
        "tool": str(tool or ""),
        "status": str(status or ""),
    }
    if started_at:
        event["startedAt"] = str(started_at)
    if ended_at:
        event["endedAt"] = str(ended_at)
    if duration_ms is not None:
        event["durationMs"] = int(duration_ms)
    if input_summary is not None:
        event["inputSummary"] = _safe_summary(input_summary)
    if output_summary is not None:
        event["outputSummary"] = _safe_summary(output_summary)
    if error:
        event["error"] = str(error)
    if isinstance(meta, dict) and meta:
        event["meta"] = meta

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


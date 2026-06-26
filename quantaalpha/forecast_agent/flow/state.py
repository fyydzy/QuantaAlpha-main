"""Flow state and checkpoint persistence for forecast agent runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from quantaalpha.forecast_agent.flow.stages import FlowStage, normalize_flow_stage

FLOW_CHECKPOINT_FILENAME = "flow_checkpoint.json"


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class FlowState:
    """Minimal flow state persisted for resume/inspection."""

    task_id: str
    status: str = "running"
    stage: str = FlowStage.PARSE_INTENT.value
    updated_at: str = field(default_factory=_now_iso)
    last_checkpoint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_checkpoint(
    *,
    task_id: str,
    stage: str | FlowStage,
    status: str,
    payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "taskId": task_id,
        "stage": normalize_flow_stage(stage),
        "status": str(status or "running"),
        "timestamp": _now_iso(),
    }
    if isinstance(payload, dict):
        item["payload"] = payload
    if error:
        item["error"] = str(error)
    return item


def checkpoint_path(out_root: str | Path) -> Path:
    root = Path(out_root)
    return root / FLOW_CHECKPOINT_FILENAME


def write_checkpoint(out_root: str | Path, checkpoint: dict[str, Any]) -> Path:
    path = checkpoint_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_checkpoint(out_root: str | Path) -> dict[str, Any] | None:
    path = checkpoint_path(out_root)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


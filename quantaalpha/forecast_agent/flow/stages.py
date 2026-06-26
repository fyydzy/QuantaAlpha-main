"""Flow stage constants for forecast agent runtime."""

from __future__ import annotations

from enum import Enum


class FlowStage(str, Enum):
    PARSE_INTENT = "parse_intent"
    CONFIRM_INTENT = "confirm_intent"
    INTENT_APPLIED = "intent_applied"
    DATA_LOADING = "data_loading"
    DIAGNOSE = "diagnose"
    RECOMMEND = "recommend"
    CONFIRM_FEATURES = "confirm_features"
    FEATURES_APPLIED = "features_applied"
    COMPARE = "compare"
    QA = "qa"
    COMPLETED = "completed"
    FAILED = "failed"


def normalize_flow_stage(stage: str | FlowStage) -> str:
    """Normalize stage values to string."""
    if isinstance(stage, FlowStage):
        return stage.value
    text = str(stage or "").strip()
    return text or FlowStage.PARSE_INTENT.value


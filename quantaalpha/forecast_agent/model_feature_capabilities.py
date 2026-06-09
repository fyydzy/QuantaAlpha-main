"""Per-model feature_set rules for quantaalpha forecast (not forecast_search)."""

from __future__ import annotations

import warnings
from typing import Iterable

from quantaalpha.forecast_agent.feature_engineering import SARIMAX_EXOG_COLS
from quantaalpha.forecast_agent.lstm_agent import SELECTED_FEATURES

TABULAR_MODELS = frozenset(
    {
        "lasso",
        "elasticnet",
        "ridge",
        "xgboost",
        "lightgbm",
        "catboost",
        "random_forest",
    }
)

SARIMAX_MODEL = "sarimax"
LSTM_MODEL = "lstm"
TIMESFM_MODEL = "timesfm"


def resolve_model_feature_set(model_name: str, feature_set: list[str] | None) -> list[str]:
    """Normalize YAML/CLI feature_set for a backend before attaching to ForecastTask."""
    model = model_name.strip().lower()
    raw = [str(x) for x in (feature_set or []) if str(x).strip()]

    if model in TABULAR_MODELS:
        return raw

    if model == SARIMAX_MODEL:
        if not raw:
            return []
        allowed = set(SARIMAX_EXOG_COLS)
        picked = [c for c in raw if c in allowed]
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"模型 {model} 的 model_features 含不支持的外生列 {unknown}；"
                f"允许: {list(SARIMAX_EXOG_COLS)}"
            )
        if not picked:
            raise ValueError(f"模型 {model} 的 model_features 与 SARIMAX 外生列无交集")
        return picked

    if model == LSTM_MODEL:
        if not raw:
            return []
        allowed = set(SELECTED_FEATURES)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"模型 {model} 的 model_features 仅支持子集 {list(SELECTED_FEATURES)}，"
                f"未知列: {unknown}"
            )
        return raw

    if model == TIMESFM_MODEL:
        if raw:
            warnings.warn(
                f"模型 {model} 忽略 model_features（TimesFM 不使用 tabular 特征列）",
                stacklevel=2,
            )
        return []

    if raw:
        warnings.warn(f"模型 {model} 未定义 feature_set 规则，将忽略 model_features", stacklevel=2)
    return []


def preflight_tabular_feature_set(
    model_name: str,
    feature_set: list[str],
    *,
    excel_path: str,
) -> None:
    """Ensure configured columns exist in the engineered feature frame."""
    if not feature_set:
        return
    from quantaalpha.forecast_agent.data import load_tabular_feature_frame
    from quantaalpha.forecast_agent.feature_engineering import select_feature_columns

    df = load_tabular_feature_frame(excel_path)
    try:
        select_feature_columns(df, feature_set)
    except ValueError as exc:
        raise ValueError(f"模型 {model_name}: {exc}") from exc


__all__ = [
    "TABULAR_MODELS",
    "resolve_model_feature_set",
    "preflight_tabular_feature_set",
]

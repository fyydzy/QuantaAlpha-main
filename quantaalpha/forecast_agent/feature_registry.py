"""Feature registry for forecast_search.

This module defines the *selectable* feature pool for LLM planning.
It should be conservative: only include stable numeric features that are
already produced by `feature_engineering.build_features_pipeline`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str


# NOTE:
# - Keep this list aligned with `quantaalpha.forecast_agent.feature_engineering`.
# - These are tabular features. Do NOT include: date/month/gas_sales.
FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "avg_temp": FeatureSpec("avg_temp", "旬平均气温"),
    "max_temp": FeatureSpec("max_temp", "旬最高气温"),
    "min_temp": FeatureSpec("min_temp", "旬最低气温"),
    "temp_range": FeatureSpec("temp_range", "旬最高温与最低温之差"),
    "HDD": FeatureSpec("HDD", "采暖度日，表示低温造成的用气需求"),
    "extreme_cold_days": FeatureSpec("extreme_cold_days", "旬内极端寒冷日数量"),
    "time_index": FeatureSpec("time_index", "长期趋势项"),
    "month_sin": FeatureSpec("month_sin", "年度周期正弦编码"),
    "month_cos": FeatureSpec("month_cos", "年度周期余弦编码"),
    "ten_sin": FeatureSpec("ten_sin", "旬度周期正弦编码"),
    "ten_cos": FeatureSpec("ten_cos", "旬度周期余弦编码"),
    "is_heating_season": FeatureSpec("is_heating_season", "是否属于供暖季（11,12,1,2,3月）"),
    "Lag_36": FeatureSpec("Lag_36", "同比滞后销量（上一年同旬）"),
    "HDD_squared": FeatureSpec("HDD_squared", "HDD 的平方项，表示极端低温的非线性影响"),
    "HDD_cross_Lag_36": FeatureSpec("HDD_cross_Lag_36", "低温与同比销量的交互项"),
    "HDD_cross_HeatingSeason": FeatureSpec("HDD_cross_HeatingSeason", "低温与供暖季的交互项"),
    "ColdDays_cross_Lag_36": FeatureSpec("ColdDays_cross_Lag_36", "极端寒冷日与同比销量的交互项"),
    "spring_rework_peak": FeatureSpec("spring_rework_peak", "春节后复工峰值（节后第1/2/3旬=1.0/0.6/0.3）"),
    "HDD_cross_spring_rework_peak": FeatureSpec(
        "HDD_cross_spring_rework_peak",
        "低温（HDD）与春节后复工峰值的交互项",
    ),
}


def feature_pool_for_prompt(registry: dict[str, FeatureSpec] | None = None) -> str:
    """Render feature pool into a stable JSON string for prompts."""
    reg = FEATURE_REGISTRY if registry is None else registry
    payload: list[dict[str, Any]] = [
        {"name": spec.name, "description": spec.description} for spec in reg.values()
    ]
    payload.sort(key=lambda x: x["name"])
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "FeatureSpec",
    "FEATURE_REGISTRY",
    "feature_pool_for_prompt",
]


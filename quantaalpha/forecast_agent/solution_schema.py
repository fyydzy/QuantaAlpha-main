"""Solution schema for forecast_search (LLM proposes feature_set candidates).

LLM is expected to output *strict JSON* that matches this schema.
We keep the schema lightweight (no pydantic dependency) and validate in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ForecastSolution:
    solution_id: str
    name: str
    hypothesis: str
    feature_set: list[str]


@dataclass(frozen=True)
class ForecastSolutionLibrary:
    solutions: list[ForecastSolution]


def _ensure_str(x: Any, field: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"字段 {field} 必须为字符串")
    return x


def _ensure_str_list(x: Any, field: str) -> list[str]:
    if not isinstance(x, list) or not all(isinstance(i, str) for i in x):
        raise ValueError(f"字段 {field} 必须为字符串数组")
    return list(x)


def parse_forecast_solution_library(raw: str | list[dict[str, Any]] | dict[str, Any]) -> ForecastSolutionLibrary:
    """Parse and validate strict JSON into ForecastSolutionLibrary.

    Accepts:
    - a raw JSON string (expected to be an array of solutions)
    - a Python object already loaded from JSON (array or dict wrapper)
    """
    if isinstance(raw, str):
        payload = json.loads(raw)
    else:
        payload = raw

    # Accept either:
    # - [ {solution_id:..., ...}, ... ]
    # - { "solutions": [ ... ] }
    if isinstance(payload, dict) and "solutions" in payload:
        payload = payload["solutions"]

    if not isinstance(payload, list):
        raise ValueError("LLM 输出必须是 JSON 数组或包含 solutions 的 JSON 对象")

    solutions: list[ForecastSolution] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 个方案必须是对象")
        solutions.append(
            ForecastSolution(
                solution_id=_ensure_str(item.get("solution_id"), f"solutions[{idx}].solution_id"),
                name=_ensure_str(item.get("name"), f"solutions[{idx}].name"),
                hypothesis=_ensure_str(item.get("hypothesis"), f"solutions[{idx}].hypothesis"),
                feature_set=_ensure_str_list(item.get("feature_set"), f"solutions[{idx}].feature_set"),
            )
        )
    if not solutions:
        raise ValueError("solutions 数组不能为空")

    return ForecastSolutionLibrary(solutions=solutions)


def feature_set_iter(library: ForecastSolutionLibrary) -> Iterable[str]:
    """Utility: iterate over all features proposed by the library."""
    for sol in library.solutions:
        for f in sol.feature_set:
            yield f


__all__ = [
    "ForecastSolution",
    "ForecastSolutionLibrary",
    "parse_forecast_solution_library",
    "feature_set_iter",
]


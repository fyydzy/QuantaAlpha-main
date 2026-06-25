"""Built-in forecast tools backed by existing flow functions."""

from __future__ import annotations

from typing import Any, Callable


def built_in_forecast_tools() -> dict[str, Callable[..., Any]]:
    # Lazy import to avoid unnecessary module initialization cost.
    from quantaalpha.forecast_agent.gas_forecast_flow import (
        ask_flow_qa,
        diagnose_importance,
        parse_intent,
        recommend_feature_superset,
        run_compare_and_rollup,
    )

    return {
        "parse_intent": parse_intent,
        "diagnose_importance": diagnose_importance,
        "recommend_feature_superset": recommend_feature_superset,
        "run_compare_and_rollup": run_compare_and_rollup,
        "ask_flow_qa": ask_flow_qa,
    }


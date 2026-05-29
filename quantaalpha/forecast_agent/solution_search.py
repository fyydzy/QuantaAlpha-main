"""Run feature-set solution search (Step 4).

This module loops over LLM-proposed solutions and runs a fixed forecast backend
(default: xgboost) for each solution, collecting test_metrics into a leaderboard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantaalpha.forecast_agent.forecast_planning import propose_solutions
from quantaalpha.forecast_agent.runner import ForecastRunConfig, run_forecast
from quantaalpha.forecast_agent.framework import ForecastTask
from quantaalpha.log import logger


@dataclass(frozen=True)
class ForecastSearchConfig:
    goal: str
    province: str = "河北"
    as_of_month: str = "2025-06-21"
    test_start: str = "2025-11-01"
    test_end: str = "2026-03-21"
    model: str = "xgboost"  # fixed backend for fair comparison
    context_len: int = 270
    output_dir: str = "forecast_agent_output/河北"

    number_of_solutions: int = 4
    max_feature_count: int = 10
    required_features: list[str] | None = None


def load_search_config_from_yaml(yaml_path: str | Path) -> ForecastSearchConfig:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    goal = str(raw.get("goal", "")).strip()
    if not goal:
        raise ValueError("配置缺少 goal")

    forecast = raw.get("forecast") or {}
    province = str(forecast.get("province", "河北"))
    as_of_month = str(forecast.get("as_of_month", "2025-06-21"))
    test_start = str(forecast.get("test_start", "2025-11-01"))
    test_end = str(forecast.get("test_end", "2026-03-21"))
    model = str(forecast.get("model", "xgboost"))
    context_len = int(forecast.get("context_len", 270))
    output_dir = str(forecast.get("output_dir", f"forecast_agent_output/{province}"))

    search = raw.get("solution_search") or {}
    number_of_solutions = int(search.get("number_of_solutions", 4))
    max_feature_count = int(search.get("max_feature_count", 10))
    required_features = search.get("required_features")
    if required_features in (None, "", "null"):
        required_features = []
    if not isinstance(required_features, list):
        raise ValueError("solution_search.required_features 必须是列表")

    return ForecastSearchConfig(
        goal=goal,
        province=province,
        as_of_month=as_of_month,
        test_start=test_start,
        test_end=test_end,
        model=model,
        context_len=context_len,
        output_dir=output_dir,
        number_of_solutions=number_of_solutions,
        max_feature_count=max_feature_count,
        required_features=[str(x) for x in required_features],
    )


def forecast_search_from_fire(**kwargs: Any) -> dict[str, Any]:
    """Fire 入口：quantaalpha forecast_search --config configs/forecast_search.yaml"""
    config_path = kwargs.pop("config", None) or kwargs.pop("config_path", None)
    if not config_path:
        raise ValueError("forecast_search 需要 --config 指定 YAML 配置路径")

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = load_search_config_from_yaml(path)

    # allow overriding a few common keys from CLI if needed
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    if overrides:
        cfg = ForecastSearchConfig(**{**asdict(cfg), **overrides})

    result = run_feature_solution_search(cfg)

    analysis = raw.get("analysis") or {}
    if bool(analysis.get("enabled", False)):
        from quantaalpha.forecast_agent.forecast_feedback import (
            FeedbackConfig,
            generate_forecast_search_feedback,
        )

        output = raw.get("output") or {}
        feedback_result = generate_forecast_search_feedback(
            FeedbackConfig(
                output_dir=cfg.output_dir,
                goal=cfg.goal,
                leaderboard_csv=output.get("leaderboard"),
                top_k=int(analysis.get("top_k", 3)),
                feedback_json=output.get("feedback_json"),
                feedback_md=output.get("feedback_md"),
            )
        )
        result["feedback"] = feedback_result

    return result


def _solution_out_dir(base_output_dir: str, solution_id: str, backend: str) -> str:
    return str(Path(base_output_dir) / "solutions" / solution_id / backend)


def _extract_test_metrics(result: dict[str, Any]) -> dict[str, Any]:
    # single-model agents may attach `test_metrics` on result or inside summary json
    tm = result.get("test_metrics")
    if isinstance(tm, dict) and tm:
        return tm

    # best-effort: try summary_json path if present
    outputs = result.get("outputs") or {}
    summary_path = outputs.get("summary_json")
    if summary_path:
        try:
            return json.loads(Path(summary_path).read_text(encoding="utf-8")).get("test_metrics", {}) or {}
        except Exception:
            return {}
    return {}


def run_feature_solution_search(cfg: ForecastSearchConfig) -> dict[str, Any]:
    """Main entry: propose solutions with LLM, run fixed model for each, output leaderboard."""
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    solutions = propose_solutions(
        goal=cfg.goal,
        number_of_solutions=cfg.number_of_solutions,
        max_feature_count=cfg.max_feature_count,
        required_features=cfg.required_features,
    )

    results: list[dict[str, Any]] = []
    for sol in solutions:
        sol_out_dir = _solution_out_dir(cfg.output_dir, sol.solution_id, cfg.model)
        logger.info(f"[forecast_search] Running {sol.solution_id} -> {sol_out_dir}")

        # Run forecast with fixed backend; inject solution metadata into ForecastTask
        run_cfg = ForecastRunConfig(
            model=cfg.model,
            province=cfg.province,
            as_of_month=cfg.as_of_month,
            test_start=cfg.test_start,
            test_end=cfg.test_end,
            output_dir=sol_out_dir,
            context_len=cfg.context_len,
        )

        task: ForecastTask = run_cfg.to_task(Path(sol_out_dir))
        task.solution_id = sol.solution_id
        task.solution_name = sol.name
        task.hypothesis = sol.hypothesis
        task.feature_set = list(sol.feature_set)

        # Call the selected backend agent by using run_forecast on a patched config:
        # run_forecast() will create its own task; we bypass by calling runner internals is undesirable.
        # So we temporarily set excel_path/province on config and call the agent via runner path:
        # We reuse `run_forecast` but need it to use our task metadata -> not supported.
        #
        # Therefore: call the backend agent directly is simpler.
        # However, to keep this Step 4 minimal and stable, we call `run_forecast` and then
        # write extra metadata into the summary via backend changes (xgboost reads task.feature_set).
        #
        # Here we just call the backend agent directly through runner factories.
        from quantaalpha.forecast_agent.runner import _build_model_factories, _config_to_namespace  # type: ignore

        factories = _build_model_factories(_config_to_namespace(run_cfg))
        agent = factories[cfg.model]
        result = agent.run(task)

        test_metrics = _extract_test_metrics(result)
        results.append(
            {
                "solution_id": sol.solution_id,
                "name": sol.name,
                "hypothesis": sol.hypothesis,
                "model": cfg.model,
                "features": ",".join(sol.feature_set),
                "MAPE": test_metrics.get("MAPE"),
                "RMSE": test_metrics.get("RMSE"),
                "MAE": test_metrics.get("MAE"),
                "R2": test_metrics.get("R2"),
                "output_dir": sol_out_dir,
            }
        )

    df = pd.DataFrame(results)
    if "MAPE" in df.columns:
        df = df.sort_values(by=["MAPE"], ascending=True, na_position="last").reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)

    leaderboard_path = out_root / "solution_leaderboard.csv"
    df.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")

    library_path = out_root / "solution_library.json"
    library_payload = {
        "goal": cfg.goal,
        "fixed_model": cfg.model,
        "solutions": [asdict(s) for s in solutions],
        "leaderboard_csv": str(leaderboard_path),
    }
    library_path.write_text(json.dumps(library_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "goal": cfg.goal,
        "fixed_model": cfg.model,
        "output_dir": str(out_root),
        "leaderboard_csv": str(leaderboard_path),
        "solution_library_json": str(library_path),
        "solutions": [asdict(s) for s in solutions],
    }


__all__ = [
    "ForecastSearchConfig",
    "load_search_config_from_yaml",
    "run_feature_solution_search",
    "forecast_search_from_fire",
]


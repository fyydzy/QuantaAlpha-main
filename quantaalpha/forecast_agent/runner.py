"""Forecast Agent 统一编排入口（CLI / Web / 未来 LLM 共用）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quantaalpha.forecast_agent.catboost_agent import AutoCatboostForecastAgent
from quantaalpha.forecast_agent.data import AS_OF_DATE, TEST_END, TEST_START
from quantaalpha.forecast_agent.elasticnet_agent import AutoElasticnetForecastAgent
from quantaalpha.forecast_agent.framework import ForecastTask
from quantaalpha.forecast_agent.lasso_agent import AutoLassoForecastAgent
from quantaalpha.forecast_agent.lightgbm_agent import AutoLightgbmForecastAgent
from quantaalpha.forecast_agent.lstm_agent import AutoLstmForecastAgent
from quantaalpha.forecast_agent.random_forest_agent import AutoRandomForestForecastAgent
from quantaalpha.forecast_agent.ridge_agent import AutoRidgeForecastAgent
from quantaalpha.forecast_agent.sarimax_agent import AutoSarimaxForecastAgent
from quantaalpha.forecast_agent.timesfm_agent import AutoTimesFmForecastAgent
from quantaalpha.forecast_agent.xgboost_agent import AutoXgboostForecastAgent

DEFAULT_CANDIDATE_MODELS = (
    "sarimax,lasso,elasticnet,ridge,lstm,xgboost,lightgbm,catboost,random_forest"
)


def _parse_model_features(raw: Any) -> dict[str, list[str] | None]:
    """Parse YAML model_features: model -> list[str] | None."""
    if raw in (None, "", "null"):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("model_features 必须是对象（model 名 -> 特征列列表或 null）")
    out: dict[str, list[str] | None] = {}
    for key, val in raw.items():
        model = str(key).strip().lower()
        if val in (None, "null"):
            out[model] = None
        elif isinstance(val, list):
            cols = [str(x).strip() for x in val if str(x).strip()]
            out[model] = cols if cols else None
        else:
            raise ValueError(f"model_features.{key} 必须是列表或 null")
    return out


def _normalize_candidates(raw: str) -> list[str]:
    candidates = [item.strip().lower() for item in raw.split(",") if item.strip()]
    deduped: list[str] = []
    for item in candidates:
        if item in deduped:
            continue
        deduped.append(item)
    return deduped


def _extract_test_metrics(result: dict[str, Any]) -> dict[str, float] | None:
    outputs = result.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    summary_path = outputs.get("summary_json")
    if not summary_path:
        return None
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = summary.get("test_metrics")
    return metrics if isinstance(metrics, dict) else None


def _build_model_factories(ns: Any) -> dict[str, Any]:
    return {
        "timesfm": AutoTimesFmForecastAgent(
            backend=ns.timesfm_device,
            selection_metric=ns.selection_metric,
        ),
        "sarimax": AutoSarimaxForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "lasso": AutoLassoForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "elasticnet": AutoElasticnetForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "ridge": AutoRidgeForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "lstm": AutoLstmForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "xgboost": AutoXgboostForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "lightgbm": AutoLightgbmForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "catboost": AutoCatboostForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
        "random_forest": AutoRandomForestForecastAgent(
            context_len=ns.context_len,
            selection_metric=ns.selection_metric,
        ),
    }


MODEL_FACTORY_KEYS = frozenset(
    {
        "timesfm",
        "sarimax",
        "lasso",
        "elasticnet",
        "ridge",
        "lstm",
        "xgboost",
        "lightgbm",
        "catboost",
        "random_forest",
    }
)

@dataclass
class ForecastRunConfig:
    model: str = "timesfm"
    province: str | None = None
    excel_path: str | None = None
    as_of_month: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    output_dir: str = "forecast_agent_output"
    context_len: int = 111
    selection_metric: str = "mape"
    candidate_models: str = DEFAULT_CANDIDATE_MODELS
    timesfm_device: str = "cpu"
    model_features: dict[str, list[str] | None] = field(default_factory=dict)

    def feature_set_for_model(self, model_name: str) -> list[str]:
        """Resolved feature columns for a backend (empty = agent default)."""
        from quantaalpha.forecast_agent.model_feature_capabilities import resolve_model_feature_set

        key = model_name.strip().lower()
        configured = self.model_features.get(key)
        if configured is None and key not in self.model_features:
            return []
        return resolve_model_feature_set(key, configured)

    def to_task(self, out_root: Path | None = None) -> ForecastTask:
        root = out_root if out_root is not None else Path(self.output_dir)
        return ForecastTask(
            output_dir=root,
            excel_path=Path(self.excel_path) if self.excel_path else None,
            province=self.province,
            as_of_month=self.as_of_month,
            test_start=self.test_start,
            test_end=self.test_end,
        )


def _test_metrics_selection_key(test_metrics: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(test_metrics, dict):
        return (float("inf"), float("inf"))
    mape = float(test_metrics.get("MAPE", test_metrics.get("MAPE(%)", float("inf"))))
    rmse = float(test_metrics.get("RMSE", float("inf")))
    return (mape, rmse)


def _config_to_namespace(config: ForecastRunConfig) -> Any:
    """构造与 MODEL_FACTORIES 兼容的简易 namespace。"""
    from types import SimpleNamespace

    return SimpleNamespace(
        timesfm_device=config.timesfm_device,
        selection_metric=config.selection_metric,
        context_len=config.context_len,
    )


def _run_single_model(
    model_name: str,
    config: ForecastRunConfig,
    base_task: ForecastTask,
    out_root: Path,
) -> dict[str, Any]:
    if model_name not in MODEL_FACTORY_KEYS:
        raise ValueError(f"Unsupported model: {model_name}")

    feature_set = config.feature_set_for_model(model_name)
    from quantaalpha.forecast_agent.model_feature_capabilities import (
        TABULAR_MODELS,
        preflight_tabular_feature_set,
    )

    configured = config.model_features.get(model_name.strip().lower())
    if model_name in TABULAR_MODELS and not feature_set:
        if configured is None and model_name not in config.model_features:
            feat_msg = "未配置 model_features → 默认全部 tabular 数值列"
        else:
            feat_msg = "model_features 为空/null → 默认全部 tabular 数值列"
        print(f"[forecast] {model_name}: {feat_msg}", flush=True)
    elif feature_set:
        print(f"[forecast] {model_name} 特征列 ({len(feature_set)}): {', '.join(feature_set)}", flush=True)

    if feature_set and model_name in TABULAR_MODELS:
        excel = base_task.excel_path
        if excel is None and base_task.province:
            from quantaalpha.forecast_agent.data import find_processed_excel

            excel = Path(find_processed_excel(str(base_task.province)))
        if excel is not None:
            preflight_tabular_feature_set(
                model_name,
                feature_set,
                excel_path=str(excel),
            )

    task = ForecastTask(
        output_dir=out_root / model_name,
        excel_path=base_task.excel_path,
        province=base_task.province,
        as_of_month=base_task.as_of_month,
        test_start=base_task.test_start,
        test_end=base_task.test_end,
        feature_set=list(feature_set),
    )
    factories = _build_model_factories(_config_to_namespace(config))
    agent = factories[model_name]
    result = agent.run(task)
    test_metrics = _extract_test_metrics(result)
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    summary_json = (result.get("outputs") or {}).get("summary_json")
    if summary_json:
        try:
            summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
            cols_used = summary.get("feature_cols_used")
            if cols_used:
                print(
                    f"[forecast] {model_name} 实际训练列: {', '.join(cols_used)}",
                    flush=True,
                )
                result["feature_cols_used"] = cols_used
        except Exception:
            pass
    return result


def _write_forecast_result(out_root: Path, payload: dict[str, Any]) -> Path:
    path = out_root / "forecast_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _build_standard_result(
    *,
    status: str,
    backend: str,
    best_model: str | None,
    test_metrics: dict[str, Any] | None,
    outputs: dict[str, Any],
    raw: dict[str, Any],
    errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "backend": backend,
        "best_model": best_model,
        "test_metrics": test_metrics or {},
        "outputs": outputs,
        "errors": errors or {},
        "raw": raw,
    }


def run_forecast(config: ForecastRunConfig) -> dict[str, Any]:
    """执行预测；返回标准摘要（并写入 output_dir/forecast_result.json）。"""
    out_root = Path(config.output_dir)
    task_base = config.to_task(out_root)

    if config.model == "auto":
        candidates = _normalize_candidates(config.candidate_models)
        if not candidates:
            raise ValueError("`candidate_models` 不能为空。")
        invalid = [m for m in candidates if m not in MODEL_FACTORY_KEYS]
        if invalid:
            raise ValueError(f"Unknown candidate models: {invalid}")

        all_results: list[dict[str, Any]] = []
        all_errors: dict[str, str] = {}
        best_result: dict[str, Any] | None = None
        best_test_metrics: dict[str, Any] | None = None

        for model_name in candidates:
            try:
                result = _run_single_model(model_name, config, task_base, out_root)
                all_results.append(result)
                current_test_metrics = result.get("test_metrics")
                if not isinstance(current_test_metrics, dict):
                    continue
                if best_test_metrics is None or _test_metrics_selection_key(
                    current_test_metrics,
                ) < _test_metrics_selection_key(best_test_metrics):
                    best_test_metrics = current_test_metrics
                    best_result = result
            except Exception as exc:  # noqa: BLE001
                all_errors[model_name] = str(exc)

        if best_result is None:
            payload = _build_standard_result(
                status="failed",
                backend="auto_select",
                best_model=None,
                test_metrics=None,
                outputs={},
                raw={"candidates": all_results, "errors": all_errors},
                errors=all_errors,
            )
            _write_forecast_result(out_root, payload)
            raise RuntimeError(f"All candidate models failed: {all_errors}")

        selection_payload = {
            "backend": "auto_select",
            "selection_metric": "test_metrics.MAPE",
            "best_model": best_result.get("backend"),
            "best_result": best_result,
            "candidates": all_results,
            "errors": all_errors,
        }
        summary_path = out_root / "model_selection_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        selection_payload["outputs"] = {"selection_summary_json": str(summary_path)}

        standard = _build_standard_result(
            status="ok",
            backend="auto_select",
            best_model=str(best_result.get("backend")),
            test_metrics=best_test_metrics,
            outputs={
                "selection_summary_json": str(summary_path),
                **(best_result.get("outputs") or {}),
            },
            raw=selection_payload,
            errors=all_errors,
        )
        _write_forecast_result(out_root, standard)
        return selection_payload

    result = _run_single_model(config.model, config, task_base, out_root)
    standard = _build_standard_result(
        status="ok",
        backend=str(result.get("backend", config.model)),
        best_model=str(result.get("backend", config.model)),
        test_metrics=result.get("test_metrics"),
        outputs=result.get("outputs") or {},
        raw=result,
    )
    _write_forecast_result(out_root, standard)
    return result


def load_forecast_summary(output_dir: str | Path, model: str | None = None) -> dict[str, Any]:
    """读取已产出的摘要 JSON（auto 或单模型）。"""
    root = Path(output_dir)
    result_path = root / "forecast_result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    if model == "auto" or (model is None and (root / "model_selection_summary.json").exists()):
        path = root / "model_selection_summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    if model and model != "auto":
        path = root / model / f"{model}_best_summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"未找到预测摘要: {output_dir}")


def load_forecast_curve_csv(output_dir: str | Path, model: str) -> list[dict[str, Any]]:
    """读取预测曲线 CSV（ds, yhat）供前端绘图。"""
    import pandas as pd

    csv_path = Path(output_dir) / model / f"{model}_forecast.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append({"ds": str(row.get("ds", "")), "yhat": float(row.get("yhat", 0))})
    return records


def load_config_from_yaml(yaml_path: str | Path) -> ForecastRunConfig:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("data") or {}
    province = data.get("province") or os.environ.get("FORECAST_DEFAULT_PROVINCE")
    excel_path = data.get("excel_path")
    if excel_path in (None, "null", ""):
        excel_path = None

    out_dir = raw.get("output_dir") or os.environ.get("FORECAST_OUTPUT_DIR", "forecast_agent_output")
    if province and "{province}" in str(out_dir):
        out_dir = str(out_dir).replace("{province}", str(province))
    elif province and not str(out_dir).endswith(str(province)):
        out_dir = str(Path(out_dir) / province)

    return ForecastRunConfig(
        model=str(raw.get("model", "timesfm")),
        province=province,
        excel_path=excel_path,
        as_of_month=raw.get("as_of_month", AS_OF_DATE),
        test_start=raw.get("test_start", TEST_START),
        test_end=raw.get("test_end", TEST_END),
        output_dir=str(out_dir),
        context_len=int(raw.get("context_len", 111)),
        selection_metric=str(raw.get("selection_metric", "mape")),
        candidate_models=str(raw.get("candidate_models", DEFAULT_CANDIDATE_MODELS)),
        timesfm_device=str(raw.get("timesfm_device", "cpu")),
        model_features=_parse_model_features(raw.get("model_features")),
    )


def merge_config_with_overrides(
    base: ForecastRunConfig,
    overrides: dict[str, Any],
) -> ForecastRunConfig:
    """命令行 / Fire 参数覆盖 YAML。"""
    data = asdict(base)
    key_map = {
        "excel": "excel_path",
        "as_of_month": "as_of_month",
        "test_start": "test_start",
        "test_end": "test_end",
        "output_dir": "output_dir",
        "context_len": "context_len",
        "selection_metric": "selection_metric",
        "candidate_models": "candidate_models",
        "timesfm_device": "timesfm_device",
        "model": "model",
        "province": "province",
        "model_features": "model_features",
    }
    for key, value in overrides.items():
        if value is None:
            continue
        field_name = key_map.get(key, key)
        if field_name in data:
            data[field_name] = value
    return ForecastRunConfig(**data)


_FIRE_KEY_MAP = {
    "as_of_month": "as_of_month",
    "test_start": "test_start",
    "test_end": "test_end",
    "output_dir": "output_dir",
    "context_len": "context_len",
    "selection_metric": "selection_metric",
    "candidate_models": "candidate_models",
    "timesfm_device": "timesfm_device",
    "model": "model",
    "province": "province",
    "excel": "excel_path",
    "config": "config_path",
}


def _default_forecast_config_path() -> str | None:
    """Best-effort default config for `quantaalpha forecast` Fire entry."""
    candidates = [
        Path.cwd() / "configs" / "forecast.yaml",
        Path(__file__).resolve().parents[2] / "configs" / "forecast.yaml",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def forecast_from_fire(**kwargs: Any) -> dict[str, Any]:
    """Google Fire 入口：quantaalpha forecast ..."""
    config_path = (
        kwargs.pop("config", None)
        or kwargs.pop("config_path", None)
        or _default_forecast_config_path()
    )
    if config_path:
        base = load_config_from_yaml(config_path)
    else:
        base = ForecastRunConfig()

    overrides: dict[str, Any] = {}
    for fire_key, field_name in _FIRE_KEY_MAP.items():
        if fire_key in kwargs and kwargs[fire_key] is not None:
            overrides[field_name if field_name != "config_path" else fire_key] = kwargs[fire_key]

    for k, v in kwargs.items():
        if k not in _FIRE_KEY_MAP and v is not None:
            overrides[k] = v

    config = merge_config_with_overrides(base, overrides)
    return run_forecast(config)


__all__ = [
    "ForecastRunConfig",
    "DEFAULT_CANDIDATE_MODELS",
    "run_forecast",
    "load_forecast_summary",
    "load_forecast_curve_csv",
    "load_config_from_yaml",
    "merge_config_with_overrides",
    "forecast_from_fire",
]

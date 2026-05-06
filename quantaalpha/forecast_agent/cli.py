from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from quantaalpha.forecast_agent.framework import ForecastTask
from quantaalpha.forecast_agent.catboost_agent import AutoCatboostForecastAgent
from quantaalpha.forecast_agent.lasso_agent import AutoLassoForecastAgent
from quantaalpha.forecast_agent.lightgbm_agent import AutoLightgbmForecastAgent
from quantaalpha.forecast_agent.sarimax_agent import AutoSarimaxForecastAgent
from quantaalpha.forecast_agent.timesfm_agent import AutoTimesFmForecastAgent
from quantaalpha.forecast_agent.xgboost_agent import AutoXgboostForecastAgent


ModelFactory = Callable[[argparse.Namespace], Any]
MODEL_FACTORIES: dict[str, ModelFactory] = {
    "timesfm": lambda args: AutoTimesFmForecastAgent(
        backend=args.timesfm_device,
        selection_metric=args.selection_metric,
    ),
    "sarimax": lambda args: AutoSarimaxForecastAgent(
        context_len=args.context_len,
        selection_metric=args.selection_metric,
    ),
    "lasso": lambda args: AutoLassoForecastAgent(
        context_len=args.context_len,
        selection_metric=args.selection_metric,
    ),
    "xgboost": lambda args: AutoXgboostForecastAgent(
        context_len=args.context_len,
        selection_metric=args.selection_metric,
    ),
    "lightgbm": lambda args: AutoLightgbmForecastAgent(
        context_len=args.context_len,
        selection_metric=args.selection_metric,
    ),
    "catboost": lambda args: AutoCatboostForecastAgent(
        context_len=args.context_len,
        selection_metric=args.selection_metric,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="月度 Excel 燃气数据预测（预留多模型入口）。",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "timesfm", "sarimax", "lasso", "xgboost", "lightgbm", "catboost"],
        default="timesfm",
        help="模型后端名称；auto 为多模型自动比选。",
    )
    parser.add_argument(
        "--candidate-models",
        default="sarimax,lasso,xgboost,lightgbm,catboost",
        help="仅在 --model auto 时生效，逗号分隔的候选模型列表。",
    )
    parser.add_argument("--excel", default=None, help="Excel 路径（与 --province 二选一）")
    parser.add_argument(
        "--province",
        default=None,
        help="省份名，用于查找 data/processed_data/{省份}.xlsx",
    )
    parser.add_argument(
        "--as-of-month",
        default=None,
        help="训练截止月 YYYY-MM；缺省见 data.AS_OF_MONTH",
    )
    parser.add_argument("--test-start", default=None, help="测试起始 YYYY-MM")
    parser.add_argument("--test-end", default=None, help="测试结束 YYYY-MM")
    parser.add_argument(
        "--output-dir",
        default="forecast_agent_output",
        help="输出根目录；单模型写入 {output_dir}/{model}，auto 会为每个候选模型分别建子目录并产出汇总文件",
    )
    parser.add_argument(
        "--timesfm-device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="TimesFM 推理设备",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mape", "score"],
        default="mape",
        help="选参指标：mape 或 0.7*SMAPE+0.3*RMSE 的 score",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=111,
        help="SARIMAX/Lasso 使用的固定历史窗口长度；可先由 TimesFM 实验结果确定后传入",
    )
    return parser


def _selection_key(feedback: dict[str, Any], metric: str) -> tuple[float, float]:
    primary = float(feedback.get(metric, float("inf")))
    secondary_name = "score" if metric == "mape" else "mape"
    secondary = float(feedback.get(secondary_name, float("inf")))
    return (primary, secondary)


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


def _test_metrics_selection_key(test_metrics: dict[str, Any] | None) -> tuple[float, float]:
    if not isinstance(test_metrics, dict):
        return (float("inf"), float("inf"))
    mape = float(test_metrics.get("MAPE(%)", float("inf")))
    rmse = float(test_metrics.get("RMSE", float("inf")))
    return (mape, rmse)


def _run_single_model(
    model_name: str,
    args: argparse.Namespace,
    base_task: ForecastTask,
    out_root: Path,
) -> dict[str, Any]:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unsupported model: {model_name}")

    task = ForecastTask(
        output_dir=out_root / model_name,
        excel_path=base_task.excel_path,
        province=base_task.province,
        as_of_month=base_task.as_of_month,
        test_start=base_task.test_start,
        test_end=base_task.test_end,
    )
    agent = MODEL_FACTORIES[model_name](args)
    result = agent.run(task)
    test_metrics = _extract_test_metrics(result)
    if test_metrics is not None:
        result["test_metrics"] = test_metrics
    return result


def _normalize_candidates(raw: str) -> list[str]:
    candidates = [item.strip().lower() for item in raw.split(",") if item.strip()]
    deduped: list[str] = []
    for item in candidates:
        if item in deduped:
            continue
        deduped.append(item)
    return deduped


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.output_dir)
    task_base = ForecastTask(
        output_dir=out_root,
        excel_path=Path(args.excel) if args.excel else None,
        province=args.province,
        as_of_month=args.as_of_month,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    if args.model == "auto":
        candidates = _normalize_candidates(args.candidate_models)
        if not candidates:
            raise ValueError("`--candidate-models` 不能为空。")

        invalid = [m for m in candidates if m not in MODEL_FACTORIES]
        if invalid:
            raise ValueError(f"Unknown candidate models: {invalid}")

        all_results: list[dict[str, Any]] = []
        all_errors: dict[str, str] = {}
        best_result: dict[str, Any] | None = None
        best_test_metrics: dict[str, Any] | None = None

        for model_name in candidates:
            try:
                result = _run_single_model(model_name, args, task_base, out_root)
                all_results.append(result)
                current_test_metrics = result.get("test_metrics")
                if not isinstance(current_test_metrics, dict):
                    continue
                if best_test_metrics is None or _test_metrics_selection_key(
                    current_test_metrics,
                ) < _test_metrics_selection_key(best_test_metrics):
                    best_test_metrics = current_test_metrics
                    best_result = result
            except Exception as exc:  # noqa: BLE001 - 聚合模式容错
                all_errors[model_name] = str(exc)

        if best_result is None:
            raise RuntimeError(f"All candidate models failed: {all_errors}")

        selection_payload = {
            "backend": "auto_select",
            "selection_metric": "test_metrics.MAPE(%)",
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
        selection_payload["outputs"] = {
            "selection_summary_json": str(summary_path),
        }
        print(json.dumps(selection_payload, ensure_ascii=False, indent=2))
    else:
        result = _run_single_model(args.model, args, task_base, out_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

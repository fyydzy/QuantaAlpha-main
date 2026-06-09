from __future__ import annotations

import argparse
import json
from typing import Any

from quantaalpha.forecast_agent.runner import (
    DEFAULT_CANDIDATE_MODELS,
    ForecastRunConfig,
    MODEL_FACTORY_KEYS,
    load_config_from_yaml,
    merge_config_with_overrides,
    run_forecast,
)

# 向后兼容：外部可继续 from quantaalpha.forecast_agent.cli import MODEL_FACTORIES
MODEL_FACTORIES = MODEL_FACTORY_KEYS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="旬度 Excel 燃气数据预测（预留多模型入口）。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件路径（如 configs/forecast.yaml）；命令行参数覆盖 YAML",
    )
    parser.add_argument(
        "--model",
        choices=["auto", *sorted(MODEL_FACTORY_KEYS)],
        default=None,
        help="模型后端名称；auto 为多模型自动比选。",
    )
    parser.add_argument(
        "--candidate-models",
        default=None,
        help=f"仅在 --model auto 时生效。默认: {DEFAULT_CANDIDATE_MODELS}",
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
        help="训练截止旬开始日 YYYY-MM-DD（如 2025-06-21）；缺省见 data.AS_OF_DATE",
    )
    parser.add_argument("--test-start", default=None, help="测试起始旬开始日 YYYY-MM-DD")
    parser.add_argument("--test-end", default=None, help="测试结束旬开始日 YYYY-MM-DD")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出根目录；单模型写入 {output_dir}/{model}，auto 会为每个候选模型分别建子目录",
    )
    parser.add_argument(
        "--timesfm-device",
        choices=["cpu", "gpu"],
        default=None,
        help="TimesFM 推理设备",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mape", "score"],
        default=None,
        help="选参指标：mape 或 0.7*SMAPE+0.3*RMSE 的 score",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="各模型使用的固定历史窗口长度（旬数，非月数）",
    )
    parser.add_argument(
        "--model-features-json",
        default=None,
        help="JSON 字符串，覆盖 YAML 的 model_features（例如: {\"lasso\":[\"HDD\",\"Lag_36\"]}）",
    )
    return parser


def _args_to_overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "model": args.model,
        "province": args.province,
        "excel_path": args.excel,
        "as_of_month": args.as_of_month,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "output_dir": args.output_dir,
        "context_len": args.context_len,
        "selection_metric": args.selection_metric,
        "candidate_models": args.candidate_models,
        "timesfm_device": args.timesfm_device,
    }
    if args.model_features_json:
        try:
            mapping["model_features"] = json.loads(args.model_features_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--model-features-json 不是合法 JSON: {exc}") from exc
    return {k: v for k, v in mapping.items() if v is not None}


def main() -> None:
    args = build_parser().parse_args()
    if args.config:
        base = load_config_from_yaml(args.config)
    else:
        base = ForecastRunConfig()
    config = merge_config_with_overrides(base, _args_to_overrides(args))
    result = run_forecast(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

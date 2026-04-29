from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantaalpha.forecast_agent.framework import ForecastTask
from quantaalpha.forecast_agent.timesfm_agent import AutoTimesFmForecastAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="月度 Excel 燃气数据预测（预留多模型入口）。",
    )
    parser.add_argument(
        "--model",
        choices=["timesfm", "sarimax"],
        default="timesfm",
        help="模型后端名称。",
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
    parser.add_argument("--output-dir", default="forecast_agent_output", help="输出目录")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task = ForecastTask(
        output_dir=Path(args.output_dir),
        excel_path=Path(args.excel) if args.excel else None,
        province=args.province,
        as_of_month=args.as_of_month,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    if args.model == "timesfm":
        agent = AutoTimesFmForecastAgent(
            backend=args.timesfm_device,
            selection_metric=args.selection_metric,
        )
    elif args.model == "sarimax":
        from quantaalpha.forecast_agent.sarimax_agent import AutoSarimaxForecastAgent

        agent = AutoSarimaxForecastAgent(selection_metric=args.selection_metric)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    result = agent.run(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantaalpha.forecast_agent.framework import ForecastTask
from quantaalpha.forecast_agent.sarimax_agent import AutoSarimaxForecastAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast agent for test.csv using auto-tuned SARIMAX.")
    parser.add_argument("--csv", default="test.csv", help="CSV path, default is test.csv")
    parser.add_argument("--horizon", type=int, default=100, help="Forecast horizon in days")
    parser.add_argument("--validation-days", type=int, default=180, help="Validation window in days")
    parser.add_argument("--max-loops", type=int, default=6, help="Agent search loops")
    parser.add_argument("--beam-width", type=int, default=4, help="Candidates per evolution step")
    parser.add_argument("--output-dir", default="forecast_agent_output", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task = ForecastTask(
        csv_path=Path(args.csv),
        horizon=args.horizon,
        validation_days=args.validation_days,
        output_dir=Path(args.output_dir),
    )
    agent = AutoSarimaxForecastAgent(max_loops=args.max_loops, beam_width=args.beam_width)
    result = agent.run(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

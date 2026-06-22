"""将 NetCDF 参考点 6 小时序列导出为 CSV（按时刻汇总 4 个温度指标）。

用法:
  uv run python scripts/weather/preview_nc.py
  uv run python scripts/weather/preview_nc.py --input data/weather/ecmwf_s5_hebei_t2m_20260501.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nc_io import load_point_t2m_df, open_weather_nc  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = PROJECT_ROOT / "data" / "weather"


def main() -> None:
    parser = argparse.ArgumentParser(description="NetCDF → 参考点 CSV")
    parser.add_argument("--input", type=Path, default=None, help="NetCDF；默认选 data/weather 下最新文件")
    parser.add_argument("--output", type=Path, default=None, help="输出 CSV；默认 <nc名>_preview.csv")
    args = parser.parse_args()

    if args.input:
        nc_path = args.input
    else:
        candidates = sorted(WEATHER_DIR.glob("ecmwf_s5_hebei_t2m_*.nc"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit(f"未找到 {WEATHER_DIR}/ecmwf_s5_hebei_t2m_*.nc")
        nc_path = candidates[-1]
        print(f"自动选用: {nc_path}")

    out_csv = args.output or (WEATHER_DIR / f"{nc_path.stem}_preview.csv")

    with open_weather_nc(nc_path) as ds:
        df = load_point_t2m_df(ds)

    member_col = next((c for c in ("number", "realization", "ensemble_member") if c in df.columns), None)
    if member_col:
        preview = (
            df.groupby("valid_time_bj")["t2m_c"]
            .agg(
                temp_mean_c="mean",
                temp_p10_c=lambda x: x.quantile(0.10),
                temp_p50_c=lambda x: x.quantile(0.50),
                temp_p90_c=lambda x: x.quantile(0.90),
            )
            .reset_index()
        )
        preview["member_count"] = int(df[member_col].nunique())
    else:
        preview = (
            df.groupby("valid_time_bj", as_index=False)["t2m_c"]
            .mean()
            .rename(columns={"t2m_c": "temp_mean_c"})
        )
        preview["temp_p10_c"] = preview["temp_mean_c"]
        preview["temp_p50_c"] = preview["temp_mean_c"]
        preview["temp_p90_c"] = preview["temp_mean_c"]
        preview["member_count"] = 1

    preview["valid_time_bj"] = pd.to_datetime(preview["valid_time_bj"])
    preview["valid_time_utc"] = preview["valid_time_bj"] - pd.Timedelta(hours=8)
    preview["date_bj"] = preview["valid_time_bj"].dt.strftime("%Y-%m-%d")
    preview["hour_bj"] = preview["valid_time_bj"].dt.hour
    preview = preview[
        [
            "valid_time_utc",
            "valid_time_bj",
            "date_bj",
            "hour_bj",
            "temp_mean_c",
            "temp_p10_c",
            "temp_p50_c",
            "temp_p90_c",
            "member_count",
        ]
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已保存: {out_csv.resolve()} ({len(preview)} 行)")


if __name__ == "__main__":
    main()

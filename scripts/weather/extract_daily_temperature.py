"""从 NetCDF 提取网格点日度 2m 气温（摄氏度，最近网格点、全部成员）。

用法:
  uv run python scripts/weather/extract_daily_temperature.py
  uv run python scripts/weather/extract_daily_temperature.py --input data/weather/ecmwf_s5_hebei_t2m_20260501_smoke.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nc_io import load_point_t2m_df, open_weather_nc  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_daily(input_file: Path, output_csv: Path) -> pd.DataFrame:
    with open_weather_nc(input_file) as ds:
        df = load_point_t2m_df(ds)

    df["date_bj"] = df["valid_time_bj"].dt.date
    member_col = next((c for c in ("number", "realization", "ensemble_member") if c in df.columns), None)
    if member_col:
        daily_member = (
            df.groupby(["date_bj", member_col], as_index=False)["t2m_c"]
            .mean()
            .rename(columns={"t2m_c": "daily_temp_c"})
        )
        daily_forecast = (
            daily_member.groupby("date_bj")["daily_temp_c"]
            .agg(
                temp_mean_c="mean",
                temp_p10_c=lambda x: x.quantile(0.10),
                temp_p50_c=lambda x: x.quantile(0.50),
                temp_p90_c=lambda x: x.quantile(0.90),
            )
            .reset_index()
        )
    else:
        daily_forecast = (
            df.groupby("date_bj", as_index=False)["t2m_c"]
            .mean()
            .rename(columns={"t2m_c": "temp_mean_c"})
        )
        daily_forecast["temp_p10_c"] = daily_forecast["temp_mean_c"]
        daily_forecast["temp_p50_c"] = daily_forecast["temp_mean_c"]
        daily_forecast["temp_p90_c"] = daily_forecast["temp_mean_c"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_forecast.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("已保存:", output_csv.resolve())
    print(daily_forecast.head())
    return daily_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    weather_dir = PROJECT_ROOT / "data" / "weather"
    if args.input:
        nc_path = args.input
    else:
        candidates = sorted(weather_dir.glob("ecmwf_s5_hebei_t2m_*.nc"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"未找到 {weather_dir}/ecmwf_s5_hebei_t2m_*.nc")
        nc_path = candidates[-1]
        print("自动选用:", nc_path)

    stem = nc_path.stem.replace("ecmwf_s5_hebei_t2m_", "")
    out_csv = args.output or (weather_dir / f"hebei_ecmwf_s5_daily_temperature_{stem}.csv")
    extract_daily(nc_path, out_csv)


if __name__ == "__main__":
    main()

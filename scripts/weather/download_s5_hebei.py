"""从 CDS 下载 ECMWF System 5 区域 2m 气温（NetCDF）。

用法（在项目根目录）:
  uv sync --extra weather
  uv run python scripts/weather/verify_cds_setup.py
  uv run python scripts/weather/download_s5_hebei.py --smoke   # 先试跑，只下 48h
  uv run python scripts/weather/download_s5_hebei.py             # 完整约 215 天
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cdsapi

# 项目根目录 = scripts/weather 的上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cds_env import get_cds_credentials, load_project_dotenv  # noqa: E402

DATASET = "seasonal-original-single-levels"

# 河北代表区域（石家庄附近），CDS area: [North, West, South, East]
AREA = [39.0, 113.0, 37.0, 116.0]

# 起报日（按 CDS 实际可用批次修改）
DEFAULT_YEAR = "2026"
DEFAULT_MONTH = "05"
DEFAULT_DAY = "01"


def _leadtime_hours(*, smoke: bool) -> list[str]:
    if smoke:
        # 试跑：2 天，每 6 小时一个点
        return [str(h) for h in range(6, 48 + 1, 6)]
    return [str(h) for h in range(6, 5160 + 1, 6)]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 ECMWF S5 2m temperature (NetCDF)")
    parser.add_argument("--smoke", action="store_true", help="仅下载短时效，用于验证 CDS 配置")
    parser.add_argument("--year", default=DEFAULT_YEAR)
    parser.add_argument("--month", default=DEFAULT_MONTH)
    parser.add_argument("--day", default=DEFAULT_DAY)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "weather",
        help="输出目录",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.year}{args.month}{args.day}"
    suffix = "_smoke" if args.smoke else ""
    target_file = out_dir / f"ecmwf_s5_hebei_t2m_{tag}{suffix}.nc"

    request = {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": ["2m_temperature"],
        "year": [args.year],
        "month": [args.month],
        "day": [args.day],
        "leadtime_hour": _leadtime_hours(smoke=args.smoke),
        "area": AREA,
        "data_format": "netcdf",
    }

    print("数据集:", DATASET)
    print("起报日:", f"{args.year}-{args.month}-{args.day}")
    print("区域 (N,W,S,E):", AREA)
    print("时效点数:", len(request["leadtime_hour"]))
    print("输出:", target_file.resolve())
    print("开始请求 CDS（可能排队，请勿关闭终端）...")

    load_project_dotenv(override=True)
    cds_url, cds_key = get_cds_credentials()
    if not cds_key:
        raise SystemExit(
            "未配置 CDS API。请在项目根 .env 设置 CDSAPI_URL 与 CDSAPI_KEY，见 docs/天气预测方案.md"
        )
    client = cdsapi.Client(url=cds_url, key=cds_key)
    client.retrieve(DATASET, request).download(str(target_file))
    print("下载完成:", target_file)


if __name__ == "__main__":
    main()

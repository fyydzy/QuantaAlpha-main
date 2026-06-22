"""在 Windows/中文路径下安全打开 ECMWF NetCDF（供 preview / extract 共用）。"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr

LAT = 38.04
LON = 114.51


def maybe_unzip(path: Path) -> Path:
    if zipfile.is_zipfile(path):
        unzip_dir = path.with_suffix("")
        unzip_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(unzip_dir)
        nc_files = list(unzip_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"zip 解压后没有找到 nc 文件: {unzip_dir}")
        print(f"检测到 zip，使用: {nc_files[0]}")
        return nc_files[0]
    return path


def _path_needs_temp_copy(path: Path) -> bool:
    try:
        str(path.resolve()).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


@contextmanager
def open_weather_nc(path: Path) -> Iterator[xr.Dataset]:
    path = maybe_unzip(path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    tmp_path: Path | None = None
    open_path = path
    if _path_needs_temp_copy(path):
        tmp_dir = Path(tempfile.gettempdir()) / "quantaalpha_nc"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / path.name
        shutil.copy2(path, tmp_path)
        open_path = tmp_path
        print(f"提示: 路径含非 ASCII 字符，已复制到临时文件: {open_path}")

    ds = xr.open_dataset(open_path, decode_timedelta=False)
    try:
        yield ds
    finally:
        ds.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def load_point_t2m_df(ds: xr.Dataset) -> pd.DataFrame:
    """参考点（石家庄附近）6 小时气温序列，保留全部集合成员。"""
    var_name = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars)[0]
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    target_lon = LON % 360 if float(ds[lon_name].max()) > 180 else LON

    point = ds[var_name].sel({lat_name: LAT, lon_name: target_lon}, method="nearest")
    df = point.to_dataframe(name="t2m_k").reset_index()
    df["t2m_c"] = df["t2m_k"] - 273.15

    if "valid_time" in df.columns:
        df["valid_time_utc"] = pd.to_datetime(df["valid_time"])
    else:
        init_col = "forecast_reference_time" if "forecast_reference_time" in df.columns else "time"
        step_col = "forecast_period" if "forecast_period" in df.columns else "step"
        step = df[step_col]
        delta = step if np.issubdtype(step.dtype, np.timedelta64) else pd.to_timedelta(step, unit="h")
        df["valid_time_utc"] = pd.to_datetime(df[init_col]) + delta

    df["valid_time_bj"] = df["valid_time_utc"] + pd.Timedelta(hours=8)
    return df.sort_values("valid_time_bj").reset_index(drop=True)

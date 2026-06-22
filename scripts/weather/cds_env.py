"""从项目根目录 .env 加载 ECMWF CDS 凭据。

cdsapi 原生支持环境变量 CDSAPI_URL / CDSAPI_KEY（见 cdsapi.api.get_url_key_verify）。
本模块在运行脚本时先 load_dotenv，便于与 LLM 配置放在同一 .env 文件。
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_CDS_URL = "https://cds.climate.copernicus.eu/api"


def load_project_dotenv(*, override: bool = True) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH, override=override)
    else:
        load_dotenv(override=override)


def get_cds_credentials() -> tuple[str, str]:
    """返回 (url, key)。
    """
    load_project_dotenv(override=True)
    url = (os.environ.get("CDSAPI_URL") or DEFAULT_CDS_URL).strip()
    key = (os.environ.get("CDSAPI_KEY") or "").strip()
    return url, key


def cds_config_source() -> str:
    """说明当前凭据来自哪里（用于 verify 输出）。"""
    load_project_dotenv(override=True)
    if os.environ.get("CDSAPI_KEY", "").strip():
        return str(DOTENV_PATH) if DOTENV_PATH.exists() else ".env (CDSAPI_KEY)"
    rc = Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc"))
    if rc.is_file():
        return str(rc)
    return "未配置"

"""检查 CDS API 配置是否就绪（支持 .env 或 ~/.cdsapirc，不打印完整 key）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cds_env import DEFAULT_CDS_URL, get_cds_credentials, load_project_dotenv  # noqa: E402


def main() -> None:
    load_project_dotenv(override=True)
    url, key = get_cds_credentials()

    source = "未配置"
    if key:
        from cds_env import cds_config_source

        source = cds_config_source()
    else:
        rc = Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc"))
        if rc.is_file():
            lines = rc.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                if line.strip().startswith("key:"):
                    key = line.split(":", 1)[-1].strip()
                if line.strip().startswith("url:") and not url:
                    url = line.split(":", 1)[-1].strip()
            source = str(rc)

    if not key:
        print("[FAIL] 未找到 CDS API Key")
        print("请在项目根目录 .env 中配置（推荐）：")
        print("  CDSAPI_URL=https://cds.climate.copernicus.eu/api")
        raise SystemExit(1)

    if ":" in key:
        print(
            "新版 CDS 请只填 Personal Access Token（无 UID、无冒号）"
        )
    else:
        print("[OK] 凭据格式：新版 Personal Access Token（无 UID 前缀）")

    print(f"[OK] 凭据来源: {source}")
    print(f"     url: {url or DEFAULT_CDS_URL}")
    print(f"     key: {'已设置 (' + str(len(key)) + ' 字符)'}")

    try:
        import cdsapi  # noqa: F401
    except ImportError as e:
        print(f"[FAIL] 未安装 cdsapi: {e}")
        print("请运行: uv sync --extra weather")
        raise SystemExit(1) from e
    print("[OK] cdsapi 已安装")
    print("下一步: 在 CDS 网页接受数据集条款，然后运行 download_s5_hebei.py --smoke")


if __name__ == "__main__":
    main()

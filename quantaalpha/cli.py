"""
QuantaAlpha CLI entry.

Commands:
  quantaalpha mine       - run factor mining
  quantaalpha backtest   - run backtest
  quantaalpha forecast   - run gas sales decadal forecast
  quantaalpha ui         - start log Web UI
  quantaalpha health_check - environment health check
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env (prefer project root, fallback to cwd)
_project_root = Path(__file__).resolve().parents[1]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(".env", override=True)

import fire
from quantaalpha.pipeline.factor_mining import main as mine
from quantaalpha.pipeline.factor_backtest import main as backtest
from quantaalpha.app.utils.health_check import health_check
from quantaalpha.app.utils.info import collect_info


def forecast(**kwargs):
    """旬度燃气销量预测（与 mine 并列，不接入因子挖掘循环）。"""
    from quantaalpha.forecast_agent.runner import forecast_from_fire

    return forecast_from_fire(**kwargs)


def forecast_flow(**kwargs):
    """燃气预测单轮编排流（M1）：意图解析→重要性→推荐→比选→月度汇总。"""
    from quantaalpha.forecast_agent.gas_forecast_flow import gas_forecast_flow_from_fire

    return gas_forecast_flow_from_fire(**kwargs)


def app():
    fire.Fire(
        {
            "mine": mine,
            "backtest": backtest,
            "forecast": forecast,
            "forecast_flow": forecast_flow,
            "health_check": health_check,
            "collect_info": collect_info,
        }
    )


if __name__ == "__main__":
    app()

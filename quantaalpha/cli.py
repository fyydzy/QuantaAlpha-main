"""
QuantaAlpha CLI — 燃气预测专用入口。

Commands:
  quantaalpha forecast       - 旬度燃气销量预测
  quantaalpha forecast_flow  - 预测智能体编排流（CLI）
"""

from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[1]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(".env", override=True)

import fire


def forecast(**kwargs):
    """旬度燃气销量预测。"""
    from quantaalpha.forecast_agent.runner import forecast_from_fire

    return forecast_from_fire(**kwargs)


def forecast_flow(**kwargs):
    """燃气预测单轮编排流：意图解析→重要性→推荐→比选→月度汇总。"""
    from quantaalpha.forecast_agent.gas_forecast_flow import gas_forecast_flow_from_fire

    return gas_forecast_flow_from_fire(**kwargs)


def app():
    fire.Fire(
        {
            "forecast": forecast,
            "forecast_flow": forecast_flow,
        }
    )


if __name__ == "__main__":
    app()

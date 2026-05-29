# 如何运行

## 1）准备环境

在项目根目录：

```powershell
cd E:\4.1\internship2\智能体框架\QuantaAlpha-main
$env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QUANTAALPHA = "0.1.0"

# 安装主项目 + 预测依赖（含 xgboost / torch 等）
uv sync --extra forecast
```

确保你的 LLM API Key 已写入项目根目录 `.env`（推荐）

## 2）编辑配置

修改 `configs/forecast_search.yaml`，至少确认：

- `goal`
- `forecast.province / as_of_month / test_start / test_end`
- `forecast.model`（第一版建议固定 `xgboost` 或 `random_forest`）
- `analysis.enabled: true`（自动生成总结）

## 3）一条命令运行（生成 leaderboard + feedback）

```powershell
uv run quantaalpha forecast_search --config configs/forecast_search.yaml
```

默认输出会写到 `forecast_agent_output/{province}/` 下（见 `configs/forecast_search.yaml` 的 `output.*` 配置）：

- `solution_leaderboard.csv`
- `solution_library.json`
- `forecast_feedback.json`
- `forecast_feedback.md`

## 4）只生成 feedback（不重新跑方案）

当你已经有 `solution_leaderboard.csv`（比如之前完整跑过一轮方案搜索），想只重新生成总结时：

```powershell
uv run python -c "from quantaalpha.forecast_agent.forecast_feedback import FeedbackConfig, generate_forecast_search_feedback; generate_forecast_search_feedback(FeedbackConfig(output_dir=r'forecast_agent_output/河北', goal='降低测试集 MAPE，给出更可解释的特征组合方案'))"
```

说明：

- `output_dir`：换成你实际的输出目录（一般是 `forecast_agent_output/{province}`）
- `goal`：可改成你这次希望 LLM 总结聚焦的目标

## 5）Web UI 运行命令（可视化跑方案/看榜单/点按钮只生成总结）

终端 1（后端 FastAPI）：

```powershell
uv run python frontend-v2\backend\app.py
```

终端 2（前端 Vite）：

```powershell
cd frontend-v2
npm install
npm run dev
```


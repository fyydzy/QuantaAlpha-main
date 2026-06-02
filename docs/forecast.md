# 燃气旬度预测（forecast_agent）

与 `quantaalpha mine`（因子挖掘）并列的独立子系统，用于省级燃气销量旬度预测。

## 环境说明（推荐只用 uv）

本项目使用 **uv 管理的项目虚拟环境**（`.venv`），与 Anaconda 基环境**分开**。

| 操作 | 装到 / 运行在哪 |
|------|-----------------|
| `pip install -e .`（未激活 `.venv` 时） | 容易装进 `D:\anaconda\...`，**不推荐** |
| `uv sync` + `uv run ...` | 项目 `.venv`，**推荐** |

首次使用（在项目根目录）：

```powershell
cd E:\4.1\internship2\智能体框架\QuantaAlpha-main
$env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QUANTAALPHA = "0.1.0"

# 基础依赖 + 本项目 editable 安装
uv sync

# 燃气预测（含 LSTM 所需的 torch，与 auto 候选一致）
uv sync --extra forecast
```

确认装在 `.venv`：

```powershell
uv run python -c "import sys; print(sys.executable)"
# 应输出: ...\QuantaAlpha-main\.venv\Scripts\python.exe

uv run pip show quantaalpha
# Location 应在 .venv 下
```

若曾误装进 Anaconda，可清理（可选）：

```powershell
pip uninstall quantaalpha -y
```

## 数据格式

Excel 主时间列为 `date`（旬开始日，每月 1 / 11 / 21 日），例如：

| date | gas_sales | avg_temp | HDD | ... |
|------|-----------|----------|-----|-----|
| 2016-03-01 | 15000 | 8.44 | 95.6 | ... |

默认数据路径：`data/processed_data/{省份}.xlsx`（可通过 `.env` 中 `FORECAST_DATA_DIR` 修改）。

## 命令行

**一律通过 `uv run` 调用**（不要依赖全局 `quantaalpha` 命令，除非你已在当前 shell 激活 `.venv`）。

```powershell
# 推荐：读默认配置
uv run quantaalpha forecast --config configs/forecast.yaml

# 单模型
uv run quantaalpha forecast --province 河北 --model lasso `
  --as-of-month 2025-06-21 --test-start 2025-11-01 --test-end 2026-03-21 `
  --context-len 270 --output-dir forecast_agent_output/河北

# 多模型比选（耗时较长）
uv run quantaalpha forecast --model auto --province 河北 --output-dir forecast_agent_output/河北

# 等价入口（模块方式）
uv run python -m quantaalpha.forecast_agent.cli --config configs/forecast.yaml
```

Linux / Git Bash 可用脚本（内部会优先 `quantaalpha`，否则 `python -m`）：

```bash
./run_forecast.sh --model lasso --province 河北
```

CLI 时间参数仅接受旬开始日 `YYYY-MM-DD`（如 `2025-06-21`），**不要**使用 `2025-06-3` 这类内部旬标签。

默认时间范围见 [`configs/forecast.yaml`](../configs/forecast.yaml)。

## 输出

- 单模型：`{output_dir}/{model}/` 下 `*_forecast.csv`、`*_best_summary.json`、`*_forecast_plot.png`
- `auto`：`{output_dir}/model_selection_summary.json` 与 `forecast_result.json`

## 依赖（extra）

`forecast` 一次安装 `auto` 默认候选所需的全部包（含 **torch / LSTM**）：

openpyxl、scikit-learn、lightgbm、xgboost、catboost、pmdarima、torch。

```powershell
uv sync --extra forecast
```

`forecast-all` 与 `forecast` 内容相同，仅保留旧命令兼容。

TimesFM 需按官方文档单独安装；`auto` 默认候选列表不含 TimesFM。

## Web UI（网页操作）

Web UI = 用**浏览器**填参数、看日志和图表；**不是**另一套预测算法。  
后端用 **FastAPI** 起 HTTP 服务（默认端口 8000），收到请求后在子进程里执行与 CLI 相同的 `quantaalpha.forecast_agent.cli`。

```
浏览器 :3000  →  Vite 前端  →  代理 /api  →  FastAPI :8000  →  子进程 forecast CLI
                                                      ↓
                                         forecast_agent_output/{省份}/
```

只做实验、习惯命令行时，**不必**启动 Web UI；下面步骤仅在需要网页时使用。

### 需要安装什么

| 类别 | 内容 | 说明 |
|------|------|------|
| 项目 Python | `uv sync` + `uv sync --extra forecast` | 与 CLI 相同，含 `quantaalpha` 与预测模型依赖 |
| 后端 API | fastapi、uvicorn、websockets、python-multipart、python-dotenv、pyyaml | FastAPI 服务用，装在同一 `.venv` |
| 前端 | Node.js 18+、`npm install` | 在 `frontend-v2/` 目录安装 React 依赖 |
| 数据 | `data/processed_data/{省份}.xlsx` | 与 CLI 相同 |

**一次性安装（在项目根目录执行）：**

```powershell
cd E:\4.1\internship2\智能体框架\QuantaAlpha-main
$env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QUANTAALPHA = "0.1.0"

# 1) 主项目 + 燃气预测 extra（若已执行过可跳过）
uv sync
uv sync --extra forecast

# 2) 后端 FastAPI 等（装进 .venv）
uv pip install fastapi uvicorn websockets python-multipart python-dotenv pyyaml

# 3) 前端依赖（仅首次或 package.json 变更后）
cd frontend-v2
npm install
cd ..
```

可选：复制 [`configs/.env.example`](../configs/.env.example) 为项目根 `.env`，设置 `FORECAST_DATA_DIR`、`FORECAST_DEFAULT_PROVINCE` 等；后端启动时会加载。

### 启动命令（Windows，推荐 uv）

需要**两个终端**，都在项目根或按下面 `cd` 操作。

**终端 1 — FastAPI 后端（必须用 `uv run`，保证子进程也是 `.venv` 里的 Python）：**

```powershell
cd E:\4.1\internship2\智能体框架\QuantaAlpha-main
uv run python frontend-v2\backend\app.py
```

成功后可访问：

- 健康检查：http://localhost:8000/api/health  
- API 文档：http://localhost:8000/docs  

**终端 2 — 前端开发服务器：**

```powershell
cd E:\4.1\internship2\智能体框架\QuantaAlpha-main\frontend-v2
npm run dev
```

浏览器打开 **http://localhost:3000** → 左侧 **燃气预测** → 填省份、模型、日期 → **开始预测**。

### 启动命令（Linux / macOS）

```bash
# 终端 1：后端（在项目根）
cd /path/to/QuantaAlpha-main
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QUANTAALPHA=0.1.0
uv sync --extra forecast
uv pip install fastapi uvicorn websockets python-multipart python-dotenv pyyaml
uv run python frontend-v2/backend/app.py

# 终端 2：前端
cd frontend-v2 && npm install && npm run dev
```

也可用仓库脚本（**会尝试 conda + pip**，与「只用 uv」不一致，仅作备选）：

```bash
cd frontend-v2 && bash start.sh
```

### 页面上做什么

| 表单项 | 含义 |
|--------|------|
| 省份 | 如 `河北`，对应 `data/processed_data/河北.xlsx` |
| 模型 | `auto` 多模型比选，或单选 `lstm`、`xgboost` 等 |
| 时间 | 旬开始日 `YYYY-MM-DD`（每月 1 / 11 / 21） |
| context_len | 上下文旬数，默认与 `configs/forecast.yaml` 一致（如 270） |

任务完成后，页面从 `forecast_agent_output/{省份}/` 读取指标与曲线；`auto` 与 CLI 一样可能耗时较长。

### 常见问题

| 现象 | 处理 |
|------|------|
| 页面提示后端不可用 | 确认终端 1 已启动，浏览器能打开 http://localhost:8000/api/health |
| 预测任务失败、日志缺包 | 后端必须用 `uv run python ...\app.py` 启动，并执行过 `uv sync --extra forecast` |
| LSTM 报错缺 torch | 同上，需 `forecast` extra |
| 与 CLI 结果路径不一致 | Web 默认输出目录为 `forecast_agent_output/{省份}`，与 CLI `--output-dir` 保持一致即可 |

## 程序化调用（供未来 LLM / 编排）

在已 `uv sync` 的环境中：

```python
from quantaalpha.forecast_agent.runner import ForecastRunConfig, run_forecast

result = run_forecast(ForecastRunConfig(
    model="lasso",
    province="河北",
    as_of_month="2025-06-21",
    test_start="2025-11-01",
    test_end="2026-03-21",
    output_dir="forecast_agent_output/河北",
    context_len=270,
))
```

## git命令
```bash
## 切换分支
git checkout test1
## 创建新分支
git checkout -b mine
## 推上云端
git push -u origin mine
```
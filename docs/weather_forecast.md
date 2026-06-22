# ECMWF 季节气温运行说明

**数据源**：数据集 [Seasonal forecast daily and subdaily data on single levels](https://cds.climate.copernicus.eu/)（`seasonal-original-single-levels`），2m 温度、6 小时步长，最长约 7 个月（215 天）。该数据为集合预报，每个时刻会给出 **51 套预报成员**（`number=0..50`）。当前脚本默认起报日 **2026-05-01**，区域为河北代表框（石家庄附近）。

---

## 前置条件

1. 注册并登录 [Copernicus CDS](https://cds.climate.copernicus.eu/)。
2. 在个人页复制 **Personal Access Token**（新版只有一段 key，**没有 UID**）。
3. 在数据集 **Download** 页手动接受 **Terms of Use**（未接受会导致下载权限错误）。
4. 本机已安装 [uv](https://github.com/astral-sh/uv)，并在 **项目根目录** 执行下列命令。

---

## 1. 安装依赖

```powershell
cd <项目根目录 QuantaAlpha-main>
$env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QUANTAALPHA = "0.1.0"
uv sync --extra weather
```

---

## 2. 配置 CDS 凭据

推荐在项目根 `.env` 中增加（可与 LLM 配置放在同一文件）：

```dotenv
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=粘贴网页上的_Personal_Access_Token整段
```

说明：


| 项            | 要求                                                            |
| ------------ | ------------------------------------------------------------- |
| `CDSAPI_KEY` | 直接粘贴 key                                                      |
| `CDSAPI_URL` | 使用 `https://cds.climate.copernicus.eu/api`，**不要** 写 `/api/v2` |
| 安全           | 勿将 token 提交到 Git                                              |


也可使用 `~/.cdsapirc`（格式同上 `url` + `key` 两行）；脚本 **优先读取 `.env`**。

校验配置：

```powershell
uv run python scripts/weather/verify_cds_setup.py
```

输出含 `[OK]` 后再进行下载。

---

## 3. 网页接受数据集条款

打开数据集页面 **Seasonal forecast daily and subdaily data on single levels**，进入 Download，按页面提示接受条款。每个账号、每个数据集只需做一次。

---

## 4. 下载 NetCDF

**建议先试跑**：

```powershell
uv run python scripts/weather/download_s5_hebei.py --smoke
```

确认无误后 **完整下载**（约 215 天，耗时长、可能排队）：

```powershell
uv run python scripts/weather/download_s5_hebei.py
```

常用参数：


| 参数                             | 说明       | 默认                   |
| ------------------------------ | -------- | -------------------- |
| `--smoke`                      | 仅下载短时效试跑 | 否                    |
| `--year` / `--month` / `--day` | 起报日      | `2026` / `05` / `01` |
| `--out-dir`                    | 输出目录     | `data/weather/`      |


输出文件示例：

```text
data/weather/ecmwf_s5_hebei_t2m_20260501_smoke.nc   # 试跑
data/weather/ecmwf_s5_hebei_t2m_20260501.nc         # 完整
```

下载过程中请勿关闭终端；CDS 可能长时间排队。

---

## 4.1 导出预览 CSV（可选）

将参考点 6 小时序列导出为 CSV，便于用 Excel 查看。

```powershell
uv run python scripts/weather/preview_nc.py
uv run python scripts/weather/preview_nc.py --input data/weather/ecmwf_s5_hebei_t2m_20260501.nc
```


| 参数         | 说明                                                      |
| ---------- | ------------------------------------------------------- |
| `--input`  | 指定 nc；默认用 `data/weather/` 下最新 `ecmwf_s5_hebei_t2m_*.nc` |
| `--output` | 输出 CSV；默认 `data/weather/<nc文件名>_preview.csv`            |

`preview.csv` 指标列：

- `temp_mean_c`：集合平均温度，51 套预报温度的平均值，代表“平均预测结果”
- `temp_p10_c`：10% 分位温度，偏冷情景，表示大约有 10% 的成员比它更低
- `temp_p50_c`：50% 分位温度，中位数情景，也就是比较中性的预测
- `temp_p90_c`：90% 分位温度，偏暖情景，表示大约有 90% 的成员比它更低


---

## 5. 提取日度气温 CSV

不指定输入时，自动选用 `data/weather/` 下最新的 `ecmwf_s5_hebei_t2m_*.nc`：

```powershell
uv run python scripts/weather/extract_daily_temperature.py
```

指定 NetCDF（完整下载后建议显式指定）：

```powershell
uv run python scripts/weather/extract_daily_temperature.py --input data/weather/ecmwf_s5_hebei_t2m_20260501.nc
```

可选 `--output` 自定义 CSV 路径。

输出示例：

```text
data/weather/hebei_ecmwf_s5_daily_temperature_20260501_smoke.csv
data/weather/hebei_ecmwf_s5_daily_temperature_20260501.csv
```

CSV 取最近网格点（石家庄附近，38.04°N, 114.51°E）并保留全部成员。日度输出包含：

- `temp_mean_c`：集合平均温度，51 套预报温度的平均值，代表“平均预测结果”
- `temp_p10_c`：10% 分位温度，偏冷情景，表示大约有 10% 的成员比它更低
- `temp_p50_c`：50% 分位温度，中位数情景，也就是比较中性的预测
- `temp_p90_c`：90% 分位温度，偏暖情景，表示大约有 90% 的成员比它更低

---

## 6. Web 查看（天气预测页）

启动前后端后，导航栏 **天气预测**（在 **燃气预测** 之前）可：

- 顶部展示数据源说明
- 左栏：选择 `*_preview.csv`，按北京时间日期 + 整点查看 `temp_mean_c/temp_p10_c/temp_p50_c/temp_p90_c`
- 右栏：选择 `hebei_ecmwf_s5_daily_temperature_*.csv`，按北京日期查看 `temp_mean_c/temp_p10_c/temp_p50_c/temp_p90_c`

接口：`GET /api/v1/weather/files`、`preview/meta`、`preview/value`、`daily`。

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

---

## 7. 对接燃气旬度预测（待做）

---

## 脚本


| 文件                                             | 作用                                        |
| ---------------------------------------------- | ----------------------------------------- |
| `scripts/weather/cds_env.py`                   | 从项目 `.env` 加载 `CDSAPI_URL` / `CDSAPI_KEY` |
| `scripts/weather/verify_cds_setup.py`          | 检查凭据与 `cdsapi` 是否可用                       |
| `scripts/weather/download_s5_hebei.py`         | CDS 下载 NetCDF                             |
| `scripts/weather/preview_nc.py`                | NetCDF → 参考点预览 CSV（无出图）                   |
| `scripts/weather/extract_daily_temperature.py` | NetCDF → 日度 CSV                           |



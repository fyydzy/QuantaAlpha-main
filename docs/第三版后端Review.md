# 后端Review

---

## 1. 总体分工

```text
ForecastPage.tsx  ──start/continue──▶  app.py (_run_forecast_agent)
                                           │
                                           ▼
                              gas_forecast_flow.py（编排）
                                           │
                                           ▼
                              runner.run_forecast（各模型）

ForecastPage.tsx  ──completed 后──▶  POST /forecast/qa  ──▶  forecast_qa.py
```

---

## 2. 后端：`frontend-v2/backend/app.py`

### 2.0 整条链路

```text
【A】用户在前端输入「预测2026年4月河北…」并发送          → §2.2
    │
    │  HTTP POST /api/v1/forecast/agent/start  { query, province(前端默认/高级设置), ... }
    │       ※ province 是前端兜底；写入 intent 的省份见【C】① flow.parse_intent
    ▼
【B】start_forecast_agent（~1695）                        → §2.3
    ├─ task_id = _gen_id()
    ├─ tasks[task_id] = { status:"running", metrics:{}, progress:{...} }
    ├─ asyncio.create_task(_run_forecast_agent(...))   ← 后台开跑，HTTP 立刻返回 taskId
    └── return { taskId, task }  ──► 前端 TaskContext 记下 taskId，连 WebSocket

【C】后台协程 _run_forecast_agent（~1403）               → §2.4
    │
    ├─ ① WS progress 10%
    ├─ to_thread( flow.parse_intent )                    ← 【flow】
    ├─ _emit_stage("parse_intent", intent)
    │
    ├─ ② _wait_for_continue("confirm_intent", …)       ◀━━ 确认点① → §2.6
    │     └─ await event.wait()  [协程睡在这里]
    │  HTTP POST …/continue  { message }
    │     ├─ continue_forecast_agent（~1756）
    │     ├─ flow.parse_continue_message               ← 【flow】
    │     ├─ event.set() 唤醒 wait()
    │     └─ WS 用户气泡
    ├─ flow.normalize_continue_overrides               ← 【flow】
    ├─ flow.apply_intent_overrides（有覆写时）           ← 【flow】
    ├─ _emit_stage("intent_applied")（有覆写时）
    │
    ├─ ③ WS progress 30% → to_thread( flow.diagnose_importance )  ← 【flow】
    ├─ _emit_stage("diagnose")
    ├─ ④ WS progress 50% → to_thread( flow.recommend_feature_superset )  ← 【flow】
    ├─ _emit_stage("recommend")
    │
    ├─ ⑤ _wait_for_continue("confirm_features", …)     ◀━━ 确认点② → §2.6
    │     └─ continue → flow.parse_continue_message → normalize_continue_overrides
    │
    ├─ ⑥ WS progress 75% → to_thread( flow.run_compare_and_rollup )  ← 【flow】
    ├─ _emit_stage("compare")
    ├─ ⑦ 可选 qaQuery → flow.ask_flow_qa               ← 【flow】
    ├─ task.metrics = { … }；task.status = "completed"
    └─ WS type:result

【D】完成后追问（不经 _run_forecast_agent）              → §2.7
    HTTP POST /api/v1/forecast/qa  { outputDir, model, query }
    └── forecast_qa.generate_forecast_qa_answer          （非 flow）
```

---

### 2.1


| 标记         | 含义                                                   |
| ---------- | ---------------------------------------------------- |
| `app`      | 仅 `app.py`（任务状态、WS、Event，无业务 LLM）                    |
| `**flow**` | 调用 `quantaalpha/forecast_agent/gas_forecast_flow.py` |
| `qa`       | 调用 `quantaalpha/app/utils/forecast_qa.py`（不在 flow 里） |


### 2.2 【A】前端发送


| 步   | 谁              | 做什么                                                            | flow |
| --- | -------------- | -------------------------------------------------------------- | ---- |
| A1  | `ForecastPage` | 用户输入 query；`province` 等为高级设置默认值                                | —    |
| A2  | 前端             | `POST /api/v1/forecast/agent/start` `{ query, province, ... }` | —    |


### 2.3 【B】`start_forecast_agent`（~1695）


| 步   | app.py                 | 做什么                                                                 | flow |
| --- | ---------------------- | ------------------------------------------------------------------- | ---- |
| B1  | `start_forecast_agent` | `task_id = _gen_id()`                                               | —    |
| B2  | 同上                     | `tasks[task_id] = { status:"running", metrics:{}, progress:{...} }` | —    |
| B3  | 同上                     | `asyncio.create_task(_run_forecast_agent(...))` **不 await**         | —    |
| B4  | 同上                     | HTTP 返回 `{ taskId, task }`                                          | —    |
| B5  | `TaskContext`          | 记下 taskId，连 WebSocket                                               | —    |


此后 **【C】在后台并行执行**；HTTP 已结束。

### 2.4 【C】`_run_forecast_agent`（~1403）

#### ① 阶段 0 — 意图解析


| 步    | app                                                                        | flow                    | 说明                                      |
| ---- | -------------------------------------------------------------------------- | ----------------------- | --------------------------------------- |
| C0-1 | `_set_progress` + WS `type:progress` 10%                                   | —                       | 更新进度条，告诉用户「正在理解你的预测需求」                  |
| C0-2 | `asyncio.to_thread(parse_intent, query, default_province=req.province, …)` | `**flow.parse_intent`** | LLM 从 query 抽出省份、目标月份、测试区间等，组成 `intent` |
| C0-3 | `_emit_stage("parse_intent", …, intent)`                                   | —                       | 聊天里弹出参数卡片，让用户看见「我理解成这样了」                |


#### ② 确认点① — 参数


| 步    | app                                                                                                                     | flow                                | 说明                                  |
| ---- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------- |
| C1-1 | `_wait_for_continue("confirm_intent", {intent})` 内：注册 Event、`awaiting_confirmation`、WS、`_emit_stage(need_confirm=True)` | —                                   | 进入第一个红灯：前端可回复，助手提示「请确认参数或说明要改什么」    |
| C1-2 | `await event.wait()` 挂起                                                                                                 | —                                   | 后台协程停住，**不跑**诊断/比选，专等用户说话           |
| C1-3 | 用户 `POST …/continue` → **§2.6**                                                                                         | `**flow.parse_continue_message`** 等 | 另一条 HTTP 送来用户话；LLM 判断是继续、取消，还是要改省/月 |
| C1-4 | `wait()` 返回；`approved=false` → `RuntimeError` 失败                                                                        | —                                   | 用户取消则任务失败；同意则协程醒来，往下走阶段 1           |
| C1-5 | `normalize_continue_overrides("confirm_intent", overrides)`                                                             | `**flow**`                          | 校验、归一化用户要改的字段（如省份简称、日期格式）           |
| C1-6 | 若有覆写：`apply_intent_overrides(intent, overrides)` → `_emit_stage("intent_applied")`                                      | `**flow**`                          | 把改过的参数写回 `intent`，聊天里再发一张更新后的参数卡    |


#### ③ 阶段 1 — xgboost 诊断


| 步    | app                                                             | flow                           | 说明                                        |
| ---- | --------------------------------------------------------------- | ------------------------------ | ----------------------------------------- |
| C2-1 | WS progress 30%                                                 | —                              | 进度到 30%，提示「正在跑 xgboost 特征诊断」              |
| C2-2 | `asyncio.to_thread(diagnose_importance, intent, output_dir, …)` | `**flow.diagnose_importance**` | 用**全量特征**训一版 xgboost，算出各特征重要性（还不定最终用哪些特征） |
| C2-3 | `_emit_stage("diagnose", …)`                                    | —                              | 展示 Top 重要性列表，给后面 LLM 推荐特征当证据              |


#### ④ 阶段 2 — 特征推荐


| 步    | app                                                                                 | flow                                  | 说明                           |
| ---- | ----------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------- |
| C3-1 | WS progress 50%                                                                     | —                                     | 进度到 50%，提示「LLM 在挑特征组合」       |
| C3-2 | `asyncio.to_thread(recommend_feature_superset, query, intent, importance_items, …)` | `**flow.recommend_feature_superset`** | LLM 结合用户需求 + 重要性，推荐一组要进模型的特征 |
| C3-3 | `_emit_stage("recommend", …)`                                                       | —                                     | 聊天展示推荐特征列表和理由                |


#### ⑤ 确认点② — 特征


| 步    | app                                                                                             | flow                              | 说明                            |
| ---- | ----------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------- |
| C4-1 | `_wait_for_continue("confirm_features", {featureSuperset, reason})`                             | —                                 | 第二个红灯：等人确认特征，或说想增删哪些          |
| C4-2 | 用户 continue → **§2.6**                                                                          | `**flow.parse_continue_message`** | 同确认点①，LLM 解析用户是要继续、取消，还是改特征列表 |
| C4-3 | `normalize_continue_overrides("confirm_features", overrides)`                                   | `**flow**`                        | 过滤非法特征名，只保留特征池里有的             |
| C4-4 | 若有 `featureSuperset`：app 写回 `recommend["feature_superset"]` → `_emit_stage("features_applied")` | 合并写在 app；校验在 **flow**             | 用用户确认后的特征列表替换推荐结果，并展示更新卡      |


#### ⑥ 阶段 3 — 多模型比选


| 步    | app                                                                      | flow                              | 说明                            |
| ---- | ------------------------------------------------------------------------ | --------------------------------- | ----------------------------- |
| C5-1 | WS progress 75%                                                          | —                                 | 进度到 75%，提示「多模型比选、出预测」         |
| C5-2 | `asyncio.to_thread(run_compare_and_rollup, intent, feature_superset, …)` | `**flow.run_compare_and_rollup`** | 按最终特征跑多个候选模型，比 MAPE，选最优并做月度汇总 |
| C5-3 | `_emit_stage("compare", …)`                                              | —                                 | 展示模型榜单、最优模型和汇总结果              |


#### ⑦ 可选 QA + 收尾


| 步    | app                                                    | flow                   | 说明                                                       |
| ---- | ------------------------------------------------------ | ---------------------- | -------------------------------------------------------- |
| C6-1 | 若 `req.qaQuery` 非空：`asyncio.to_thread(ask_flow_qa, …)` | `**flow.ask_flow_qa**` | （可选）启动时预填了问题，则自动对预测表问一次                                  |
| C6-2 | `task.metrics = payload`；`status="completed"`          | —                      | 把 intent、诊断、比选、`output_dir`、`selected_model` 等写入任务单，标为完成 |
| C6-3 | WS `type:result`                                       | —                      | 通知前端「整单跑完了」，可展示最终结果、开启完成后追问                              |


### 2.5 HTTP 辅助路由（2.0 未画）


| 路由                            | HTTP   | 行号    | 作用                                 | flow |
| ----------------------------- | ------ | ----- | ---------------------------------- | ---- |
| `/api/v1/forecast/agent/{id}` | GET    | ~1729 | 轮询 task；可读 `awaiting_confirmation` | —    |
| `/api/v1/forecast/agent/{id}` | DELETE | ~1736 | 取消；`event.set()` 结束 `wait()`       | —    |


### 2.6 `continue_forecast_agent`（~1756）


| 步   | app                                                              | flow       | 说明                             |
| --- | ---------------------------------------------------------------- | ---------- | ------------------------------ |
| 1   | 检查 `awaiting_confirmation`                                       | —          | 无则 **409**                     |
| 2   | `parse_continue_message(checkpoint, payload, message, …)`        | `**flow`** | LLM → `approved` + `overrides` |
| 3   | `_forecast_continue_payloads[task_id] = { approved, overrides }` | —          | 供 `wait()` 返回后读取               |
| 4   | `_forecast_continue_events[task_id].set()`                       | —          | 唤醒 §2.4 里睡着的协程                 |
| 5   | 用户气泡写入 `agent_messages` + WS                                     | —          | —                              |


### 2.7 【D】完成后追问

**不经过** `_run_forecast_agent`；任务 `completed` 后前端另开一条路。


| 步   | 谁                       | 做什么                                                      | 模块                       |
| --- | ----------------------- | -------------------------------------------------------- | ------------------------ |
| D1  | `ForecastPage`          | `POST /api/v1/forecast/qa` `{ outputDir, model, query }` | —                        |
| D2  | `ask_forecast_qa` ~1352 | 读 `{model}_test` 表                                       | app 读盘                   |
| D3  | 同上                      | `generate_forecast_qa_answer`                            | `forecast_qa.py`（非 flow） |



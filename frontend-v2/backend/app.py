"""
QuantaAlpha Backend API — 预测智能体与天气预测。
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: frontend-v2/backend/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Ensure import quantaalpha is available (when backend is started from frontend-v2 directory, repo root is not in sys.path)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DOTENV_PATH = PROJECT_ROOT / ".env"

from quantaalpha.forecast_agent.flow import FlowStage, FlowState, make_checkpoint, write_checkpoint
from quantaalpha.forecast_agent.tools import audit_path, call_forecast_tool, write_audit_event

app = FastAPI(title="QuantaAlpha API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastQaRequest(BaseModel):
    """Request to ask LLM about forecast result table."""
    outputDir: str = Field(..., description="Forecast output root directory")
    model: Optional[str] = Field(None, description="Backend model name")
    query: str = Field(..., description="User question about forecast results")
    selectedFeatures: Optional[List[str]] = Field(
        None,
        description="Frontend selected features for current model",
    )


class ForecastAgentStartRequest(BaseModel):
    """Request to start single-round forecast agent flow."""
    query: str = Field(..., description="Natural language request")
    province: Optional[str] = Field(
        None,
        description="对话未提及省份时的兜底名；输出目录以 parse_intent 解析结果为准",
    )
    candidateModels: Optional[str] = Field(None, description="Comma-separated candidate models")
    outputDir: Optional[str] = Field(
        None,
        description="可选自定义输出根目录，支持 {province} 占位；默认 forecast_agent_output/{省份}/agent_flow_{ts}",
    )
    contextLen: Optional[int] = Field(270, description="Context length")
    maxFeatureCount: Optional[int] = Field(10, description="Max recommended features")
    importanceTopK: Optional[int] = Field(12, description="Top-K feature importance for prompting")
    requiredFeatures: Optional[List[str]] = Field(None, description="Required features in recommendation")
    qaQuery: Optional[str] = Field(None, description="Optional QA query after forecast")


class ForecastAgentContinueRequest(BaseModel):
    """Request to continue a paused forecast agent checkpoint."""
    checkpoint: Optional[str] = Field(None, description="Expected waiting checkpoint")
    approved: bool = Field(True, description="Whether user approves to continue")
    overrides: Optional[Dict[str, Any]] = Field(None, description="Optional override payload")
    message: Optional[str] = Field(None, description="Optional user message")


class SystemConfigUpdate(BaseModel):
    """Partial update to system configuration (.env)."""
    QLIB_DATA_DIR: Optional[str] = None
    DATA_RESULTS_DIR: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    CHAT_MODEL: Optional[str] = None
    REASONING_MODEL: Optional[str] = None


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None

# ========================== In-Memory State ==========================

tasks: Dict[str, Dict[str, Any]] = {}
ws_connections: Dict[str, List[WebSocket]] = {}  # task_id -> list of WS
# 燃气预测智能体：人机确认暂停/续跑句柄（必须放模块级，不能放进 tasks[task_id]，否则 GET 任务 JSON 序列化失败）
_forecast_continue_events: Dict[str, asyncio.Event] = {}
_forecast_continue_payloads: Dict[str, Dict[str, Any]] = {}
_forecast_runtime_tasks: Dict[str, asyncio.Task] = {}


def _gen_id() -> str:
    return str(uuid.uuid4())[:8]


def _now() -> str:
    return datetime.now().isoformat()


def _task_for_api(task: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable copy of task state (strip runtime-only fields)."""
    return {k: v for k, v in task.items() if not str(k).startswith("_")}


def _load_dotenv_dict() -> Dict[str, str]:
    """Parse the .env file into a dict (simple key=value, ignoring comments)."""
    env: Dict[str, str] = {}
    if DOTENV_PATH.exists():
        for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, _, val = stripped.partition("=")
                env[key.strip()] = val.strip()
    return env


async def _broadcast(task_id: str, message: Dict[str, Any]):
    """Send a JSON message to all WebSocket clients for a task."""
    if task_id not in ws_connections:
        return
    dead: List[WebSocket] = []
    for ws in ws_connections[task_id]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections[task_id].remove(ws)


def _spawn_forecast_runtime(task_id: str, req: "ForecastAgentStartRequest") -> None:
    """Spawn and track one runtime task per forecast task id."""
    runtime_task = asyncio.create_task(_run_forecast_agent(task_id, req))
    _forecast_runtime_tasks[task_id] = runtime_task

    def _cleanup(_done: asyncio.Task) -> None:
        current = _forecast_runtime_tasks.get(task_id)
        if current is _done:
            _forecast_runtime_tasks.pop(task_id, None)

    runtime_task.add_done_callback(_cleanup)


# ========================== API Endpoints ==========================

@app.get("/")
async def root():
    return {"message": "QuantaAlpha API", "version": "2.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": _now()}


# ---- Weather (ECMWF CSV) endpoints ----

WEATHER_DIR = PROJECT_ROOT / "data" / "weather"

WEATHER_DATA_SOURCE = {
    "title": "Seasonal forecast daily and subdaily data on single levels",
    "dataset": "seasonal-original-single-levels",
    "url": "https://cds.climate.copernicus.eu/",
    "variable": "2m 温度",
    "step": "6 小时",
    "horizon": "最长约 7 个月（215 天）",
    "initDate": "2026-05-01",
    "area": "河北代表框（石家庄附近，AREA=[39,113,37,116]）",
    "gridNote": "下载为小区域网格（约 2×3 格点），展示时取距 (38.04°N, 114.51°E) 最近格点；温度展示按集合成员求平均",
}


def _resolve_weather_file(filename: str) -> Path:
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    path = (WEATHER_DIR / filename).resolve()
    root = WEATHER_DIR.resolve()
    if not str(path).startswith(str(root)):
        raise HTTPException(status_code=403, detail="路径不允许")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    return path


def _parse_bj_datetime(series: Any) -> Any:
    import pandas as pd

    return pd.to_datetime(series, errors="coerce")


@app.get("/api/v1/weather/files", response_model=ApiResponse)
async def list_weather_files():
    """List preview (6h) and daily CSV files under data/weather/."""
    preview_files: List[str] = []
    daily_files: List[str] = []
    if WEATHER_DIR.is_dir():
        preview_files = sorted(p.name for p in WEATHER_DIR.glob("*_preview.csv"))
        daily_files = sorted(
            p.name
            for p in WEATHER_DIR.glob("hebei_ecmwf_s5_daily_temperature_*.csv")
        )
    return ApiResponse(
        success=True,
        data={
            "previewFiles": preview_files,
            "dailyFiles": daily_files,
            "dataSource": WEATHER_DATA_SOURCE,
            "weatherDir": str(WEATHER_DIR),
        },
    )


@app.get("/api/v1/weather/preview/meta", response_model=ApiResponse)
async def weather_preview_meta(file: str = Query(..., description="*_preview.csv 文件名")):
    """Return available Beijing date/hour slots for a preview CSV."""
    import pandas as pd

    path = _resolve_weather_file(file)
    df = pd.read_csv(path)
    cols = [
        c
        for c in ("valid_time_bj", "latitude", "longitude", "number", "realization", "ensemble_member")
        if c in df.columns
    ]
    df = df[cols] if cols else df
    if "valid_time_bj" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV 缺少 valid_time_bj 列")

    ts = _parse_bj_datetime(df["valid_time_bj"])
    df = df.assign(
        date_bj=ts.dt.strftime("%Y-%m-%d"),
        hour_bj=ts.dt.hour,
    )
    slots = (
        df[["date_bj", "hour_bj"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["date_bj", "hour_bj"])
    )
    dates = sorted(slots["date_bj"].unique().tolist())
    hours_by_date: Dict[str, List[int]] = {}
    for d in dates:
        hours_by_date[d] = slots.loc[slots["date_bj"] == d, "hour_bj"].astype(int).tolist()

    lat = float(df["latitude"].iloc[0]) if "latitude" in df.columns and len(df) else None
    lon = float(df["longitude"].iloc[0]) if "longitude" in df.columns and len(df) else None
    member_col = next((c for c in ("number", "realization", "ensemble_member") if c in df.columns), None)
    members: List[int] = []
    if member_col is not None:
        members = sorted(int(x) for x in df[member_col].dropna().unique())

    return ApiResponse(
        success=True,
        data={
            "file": file,
            "dates": dates,
            "hoursByDate": hours_by_date,
            "gridLatitude": lat,
            "gridLongitude": lon,
            "rowCount": len(df),
            "members": members,
        },
    )


@app.get("/api/v1/weather/preview/value", response_model=ApiResponse)
async def weather_preview_value(
    file: str = Query(...),
    date_bj: str = Query(..., description="YYYY-MM-DD"),
    hour_bj: int = Query(..., ge=0, le=23),
    member: Optional[int] = Query(None, description="可选成员编号；不传则为集合平均"),
):
    """6h preview temperature (°C) at selected Beijing date/hour."""
    import pandas as pd

    path = _resolve_weather_file(file)
    df = pd.read_csv(path)
    if "valid_time_bj" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV 缺少 valid_time_bj")

    ts = _parse_bj_datetime(df["valid_time_bj"])
    mask = (ts.dt.strftime("%Y-%m-%d") == date_bj) & (ts.dt.hour == hour_bj)
    sub = df.loc[mask]
    if sub.empty:
        return ApiResponse(
            success=False,
            error=f"未找到 {date_bj} {hour_bj:02d}:00（北京时间）的记录",
        )

    row0 = sub.iloc[0]
    has_agg_cols = {"temp_mean_c", "temp_p10_c", "temp_p50_c", "temp_p90_c"}.issubset(set(sub.columns))
    if has_agg_cols:
        temp_mean = float(row0["temp_mean_c"])
        temp_p10 = float(row0["temp_p10_c"])
        temp_p50 = float(row0["temp_p50_c"])
        temp_p90 = float(row0["temp_p90_c"])
        member_count = int(row0["member_count"]) if "member_count" in sub.columns and not pd.isna(row0["member_count"]) else 51
    else:
        if "t2m_c" not in sub.columns:
            raise HTTPException(status_code=400, detail="CSV 缺少温度列")
        temps = sub["t2m_c"].astype(float)
        temp_mean = float(temps.mean())
        temp_p10 = float(temps.quantile(0.10))
        temp_p50 = float(temps.quantile(0.50))
        temp_p90 = float(temps.quantile(0.90))
        member_col = next((c for c in ("number", "realization", "ensemble_member") if c in sub.columns), None)
        member_count = int(sub[member_col].nunique()) if member_col else 1

    return ApiResponse(
        success=True,
        data={
            "dateBj": date_bj,
            "hourBj": hour_bj,
            "temperatureC": round(temp_mean, 2),
            "aggMode": "mean",
            "member": None,
            "memberCount": member_count,
            "sampleCount": int(len(sub)),
            "tempMinC": round(temp_p10, 2),
            "tempMaxC": round(temp_p90, 2),
            "tempP10C": round(temp_p10, 2),
            "tempP50C": round(temp_p50, 2),
            "tempP90C": round(temp_p90, 2),
            "validTimeUtc": str(row0["valid_time_utc"]) if "valid_time_utc" in sub.columns else None,
            "validTimeBj": str(row0["valid_time_bj"]) if "valid_time_bj" in sub.columns else None,
        },
    )


@app.get("/api/v1/weather/daily", response_model=ApiResponse)
async def weather_daily_data(file: str = Query(..., description="hebei_ecmwf_s5_daily_temperature_*.csv")):
    """Load daily aggregated CSV (small) for date picker."""
    import pandas as pd

    path = _resolve_weather_file(file)
    df = pd.read_csv(path)
    date_col = "date_bj" if "date_bj" in df.columns else df.columns[0]
    df[date_col] = df[date_col].astype(str)
    temp_col = "temp_mean_c" if "temp_mean_c" in df.columns else ("temp_c" if "temp_c" in df.columns else "t2m_c")
    if temp_col not in df.columns:
        raise HTTPException(status_code=400, detail="CSV 缺少 temp_c / temp_mean_c 列")

    rows = []
    for _, r in df.iterrows():
        row: Dict[str, Any] = {
            "dateBj": str(r[date_col])[:10],
            "tempMeanC": round(float(r[temp_col]), 2),
        }
        for src, key in (
            ("temp_p10_c", "tempP10C"),
            ("temp_p50_c", "tempP50C"),
            ("temp_p90_c", "tempP90C"),
        ):
            if src in df.columns and pd.notna(r.get(src)):
                row[key] = round(float(r[src]), 2)
            else:
                row[key] = row["tempMeanC"]
        rows.append(row)

    return ApiResponse(
        success=True,
        data={"file": file, "rows": rows, "tempColumn": temp_col},
    )


def _load_feature_cols_used_for_model(out_root: Path, model: str) -> list[str]:
    summary = out_root / model / f"{model}_best_summary.json"
    if not summary.exists():
        return []
    try:
        raw = json.loads(summary.read_text(encoding="utf-8"))
        cols = raw.get("feature_cols_used")
        if isinstance(cols, list):
            return [str(c) for c in cols]
    except Exception:
        return []
    return []


@app.post("/api/v1/forecast/qa", response_model=ApiResponse)
async def ask_forecast_qa(req: ForecastQaRequest):
    """Ask LLM questions based on forecast result table (pred vs actual)."""
    try:
        # Ensure .env values are visible for this process path too (not only child subprocesses).
        os.environ.update(_load_dotenv_dict())

        out = Path(req.outputDir)
        if not out.is_absolute():
            out = PROJECT_ROOT / out

        model = (req.model or "").strip().lower()
        if not model:
            raise HTTPException(status_code=400, detail="model 不能为空")

        model_dir = out / model
        table_rows = _load_forecast_test_points_for_ui(model_dir, model)
        if not table_rows:
            curve = _load_forecast_curve_for_ui(model_dir, model)
            table_rows = [
                {"ds": r.get("ds"), "period": "", "yhat": r.get("yhat"), "y": r.get("y"), "error_pct": None}
                for r in curve
                if isinstance(r, dict)
            ]
        if not table_rows:
            raise HTTPException(status_code=404, detail=f"未找到可问答的预测结果文件: {model_dir}")

        from quantaalpha.app.utils.forecast_qa import generate_forecast_qa_answer

        result = generate_forecast_qa_answer(
            query=req.query,
            rows=table_rows,
            feature_cols_used=_load_feature_cols_used_for_model(out, model),
            selected_features=list(req.selectedFeatures or []),
        )
        return ApiResponse(
            success=True,
            data={
                "answer": result.get("answer", ""),
                "modelUsed": result.get("model_used", ""),
                "rowsUsed": result.get("rows_used", 0),
                "dataMode": result.get("data_mode", "pred_only"),
                "featureColsUsed": result.get("feature_cols_used", []),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM 问答失败: {exc}") from exc


def _forecast_fallback_province(explicit: str | None = None) -> str:
    for candidate in (explicit, os.environ.get("FORECAST_DEFAULT_PROVINCE"), "河北"):
        text = str(candidate or "").strip()
        if text:
            return text
    return "河北"


def _resolve_agent_flow_out_root(
    province: str,
    *,
    run_ts: str,
    custom_output_dir: str | None = None,
) -> Path:
    """Web Agent 输出根目录：forecast_agent_output/{省份}/agent_flow_{ts}（或自定义 base + agent_flow_{ts}）。"""
    prov = _forecast_fallback_province(province)
    raw = str(custom_output_dir or "").strip()
    if raw:
        base = Path(raw.replace("{province}", prov))
    else:
        base = Path("forecast_agent_output") / prov
    if base.name.startswith("agent_flow_"):
        out = base
    else:
        out = base / f"agent_flow_{run_ts}"
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def _province_display_label(name: str) -> str:
    raw = str(name or "").strip() or "未知"
    if raw.endswith(("省", "市", "自治区", "特别行政区")):
        return raw
    if raw in {"北京", "上海", "天津", "重庆"}:
        return f"{raw}市"
    special = {
        "内蒙古": "内蒙古自治区",
        "广西": "广西壮族自治区",
        "西藏": "西藏自治区",
        "宁夏": "宁夏回族自治区",
        "新疆": "新疆维吾尔自治区",
        "香港": "香港特别行政区",
        "澳门": "澳门特别行政区",
    }
    return special.get(raw, f"{raw}省")


def _display_path_for_ui(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


async def _run_forecast_agent(
    task_id: str,
    req: ForecastAgentStartRequest,
):
    """燃气预测智能体主协程（Web 专用）。

    职责：驱动 gas_forecast_flow 各阶段、推送聊天消息、在两个人机确认点挂起等待。
    业务规则（意图/覆写/LLM prompt）在 gas_forecast_flow.py，本函数只做任务状态与 WS 广播。

    阅读顺序：_emit_stage → _wait_for_continue → 下方「阶段 0~5」顺序块 → continue_forecast_agent。
    """
    task = tasks[task_id]
    try:
        # Ensure .env credentials are visible in current process
        # (agent flow runs in-process, unlike subprocess-based tasks).
        os.environ.update(_load_dotenv_dict())

        from quantaalpha.forecast_agent.gas_forecast_flow import (
            FIXED_AS_OF_DATE,
            apply_intent_overrides,
            normalize_continue_overrides,
        )

        if not str(os.getenv("OPENAI_API_KEY", "")).strip():
            raise RuntimeError("缺少 OPENAI_API_KEY，请在项目根目录 .env 中配置后重试")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root: Path | None = None
        flow_audit_file: Path | None = None
        flow_state = FlowState(task_id=task_id)
        task["flowState"] = flow_state.to_dict()
        intent: Dict[str, Any] = {}
        diagnose: Dict[str, Any] = {}
        recommend: Dict[str, Any] = {}

        def _persist_checkpoint(
            stage: str | FlowStage,
            *,
            payload: Dict[str, Any] | None = None,
            status: str = "running",
            error: str | None = None,
        ) -> dict[str, Any]:
            cp = make_checkpoint(
                task_id=task_id,
                stage=stage,
                status=status,
                payload=payload,
                error=error,
            )
            cp_state: Dict[str, Any] = {}
            if isinstance(intent, dict) and intent:
                cp_state["intent"] = dict(intent)
            if isinstance(diagnose, dict) and diagnose:
                cp_state["diagnose"] = dict(diagnose)
            if isinstance(recommend, dict) and recommend:
                cp_state["recommend"] = dict(recommend)
            if cp_state:
                cp["state"] = cp_state
            flow_state.stage = str(cp.get("stage") or flow_state.stage)
            flow_state.status = str(cp.get("status") or flow_state.status)
            flow_state.updated_at = str(cp.get("timestamp") or _now())
            flow_state.last_checkpoint = cp
            task["flowState"] = flow_state.to_dict()
            task["lastCheckpoint"] = cp
            if out_root is not None:
                cp_file = write_checkpoint(out_root, cp)
                task["checkpointPath"] = str(cp_file)
                task["checkpointPathDisplay"] = _display_path_for_ui(str(cp_file))
            return cp

        def _sync_task_output_config(intent_province: str, resolved_out_root: Path) -> None:
            if isinstance(task.get("config"), dict):
                task["config"]["province"] = intent_province
                task["config"]["outputDir"] = str(resolved_out_root)

        def _ensure_out_root(intent_province: str) -> Path:
            nonlocal out_root, flow_audit_file
            resolved = _resolve_agent_flow_out_root(
                intent_province,
                run_ts=ts,
                custom_output_dir=req.outputDir,
            )
            resolved.mkdir(parents=True, exist_ok=True)
            out_root = resolved
            _sync_task_output_config(intent_province, resolved)
            flow_audit_file = audit_path(out_root)
            task["auditPath"] = str(flow_audit_file)
            task["auditPathDisplay"] = _display_path_for_ui(str(flow_audit_file))
            if isinstance(task.get("lastCheckpoint"), dict):
                cp_file = write_checkpoint(out_root, task["lastCheckpoint"])
                task["checkpointPath"] = str(cp_file)
                task["checkpointPathDisplay"] = _display_path_for_ui(str(cp_file))
            return resolved

        async def _run_tool(
            stage: str | FlowStage,
            tool_name: str,
            *tool_args: Any,
            input_summary: Any = None,
            meta: Dict[str, Any] | None = None,
            **tool_kwargs: Any,
        ) -> Any:
            if not isinstance(task.get("metrics"), dict):
                task["metrics"] = {}

            def _append_tool_event(event: Dict[str, Any]) -> None:
                events = task["metrics"].get("tool_events")
                if not isinstance(events, list):
                    events = []
                events.append(event)
                task["metrics"]["tool_events"] = events[-200:]
                task["metrics"]["tool_event"] = event

            started_ts = _now()
            t0 = time.perf_counter()
            try:
                result = await asyncio.to_thread(
                    call_forecast_tool,
                    tool_name,
                    *tool_args,
                    **tool_kwargs,
                )
                ended_ts = _now()
                duration_ms = int((time.perf_counter() - t0) * 1000)
                if out_root is not None:
                    file_path = write_audit_event(
                        out_root,
                        task_id=task_id,
                        stage=str(stage),
                        tool=tool_name,
                        status="ok",
                        started_at=started_ts,
                        ended_at=ended_ts,
                        duration_ms=duration_ms,
                        input_summary=input_summary,
                        output_summary=result,
                        meta=meta or {},
                    )
                    task["auditPath"] = str(file_path)
                    task["auditPathDisplay"] = _display_path_for_ui(str(file_path))
                tool_event = {
                    "stage": str(stage),
                    "tool": tool_name,
                    "status": "ok",
                    "startedAt": started_ts,
                    "endedAt": ended_ts,
                    "durationMs": duration_ms,
                }
                _append_tool_event(tool_event)
                await _broadcast(
                    task_id,
                    {
                        "type": "metrics",
                        "taskId": task_id,
                        "data": {"tool_event": tool_event},
                        "timestamp": _now(),
                    },
                )
                return result
            except Exception as exc:
                ended_ts = _now()
                duration_ms = int((time.perf_counter() - t0) * 1000)
                if out_root is not None:
                    file_path = write_audit_event(
                        out_root,
                        task_id=task_id,
                        stage=str(stage),
                        tool=tool_name,
                        status="error",
                        started_at=started_ts,
                        ended_at=ended_ts,
                        duration_ms=duration_ms,
                        input_summary=input_summary,
                        error=str(exc),
                        meta=meta or {},
                    )
                    task["auditPath"] = str(file_path)
                    task["auditPathDisplay"] = _display_path_for_ui(str(file_path))
                tool_event = {
                    "stage": str(stage),
                    "tool": tool_name,
                    "status": "error",
                    "startedAt": started_ts,
                    "endedAt": ended_ts,
                    "durationMs": duration_ms,
                    "error": str(exc),
                }
                _append_tool_event(tool_event)
                await _broadcast(
                    task_id,
                    {
                        "type": "metrics",
                        "taskId": task_id,
                        "data": {"tool_event": tool_event},
                        "timestamp": _now(),
                    },
                )
                raise

        def _set_progress(phase: str, progress: int, message: str) -> dict[str, Any]:
            task["progress"]["phase"] = phase
            task["progress"]["progress"] = progress
            task["progress"]["message"] = message
            task["progress"]["timestamp"] = _now()
            return dict(task["progress"])

        async def _emit_stage(stage: str, text: str, payload: Dict[str, Any], *, need_confirm: bool = False):
            # 往聊天流追加一条助手消息，经 WS type=metrics 推给前端（ForecastPage / TaskContext）
            agent_message = {
                "role": "assistant",
                "stage": stage,
                "messageType": f"{stage}_card",
                "text": text,
                "payload": payload,
                "needConfirm": need_confirm,
                "timestamp": _now(),
            }
            if not isinstance(task.get("metrics"), dict):
                task["metrics"] = {}
            msg_list = task["metrics"].get("agent_messages")
            if not isinstance(msg_list, list):
                msg_list = []
            msg_list.append(agent_message)
            task["metrics"]["agent_messages"] = msg_list[-200:]
            task["metrics"]["agent_message"] = agent_message
            await _broadcast(
                task_id,
                {
                    "type": "metrics",
                    "taskId": task_id,
                    "data": {
                        "agent_stage": stage,
                        "agent_message": agent_message,
                    },
                    "timestamp": _now(),
                },
            )

        async def _wait_for_continue(checkpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            # 人机确认：挂起本协程，直到用户 POST .../continue 写入 payload 并 event.set()
            # checkpoint: confirm_intent | confirm_features
            event = asyncio.Event()
            _forecast_continue_events[task_id] = event
            _forecast_continue_payloads.pop(task_id, None)
            if not isinstance(task.get("metrics"), dict):
                task["metrics"] = {}
            task["metrics"]["awaiting_confirmation"] = {
                "checkpoint": checkpoint,
                "timestamp": _now(),
                "payload": payload,
            }
            await _broadcast(
                task_id,
                {
                    "type": "metrics",
                    "taskId": task_id,
                    "data": {
                        "awaiting_confirmation": task["metrics"]["awaiting_confirmation"],
                    },
                    "timestamp": _now(),
                },
            )
            confirm_text = (
                "以上是我理解的预测参数。若无问题请直接回复「继续」；也可说明要改的省份或目标月份。"
                if checkpoint == "confirm_intent"
                else "以上是我推荐的特征组合。若同意请回复「继续」；也可直接说想增删哪些特征。"
            )
            await _emit_stage(
                checkpoint,
                confirm_text,
                payload,
                need_confirm=True,
            )
            await event.wait()
            # Guard: if cancel_forecast_agent woke us up (set event then cancelled task),
            # stop here instead of accidentally continuing to the next flow stage.
            if task.get("status") == "cancelled":
                raise asyncio.CancelledError()
            cont = _forecast_continue_payloads.pop(task_id, None)
            _forecast_continue_events.pop(task_id, None)
            task["metrics"]["awaiting_confirmation"] = None
            await _broadcast(
                task_id,
                {
                    "type": "metrics",
                    "taskId": task_id,
                    "data": {"awaiting_confirmation": None},
                    "timestamp": _now(),
                },
            )
            return cont if isinstance(cont, dict) else {}

        # --- 阶段 0：意图解析 + 确认点①（参数）---
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("parsing", 10, "解析用户意图..."), "timestamp": _now()})
        intent = await _run_tool(
            FlowStage.PARSE_INTENT,
            "parse_intent",
            req.query,
            input_summary={"query": req.query, "defaultProvince": _forecast_fallback_province(req.province)},
            default_province=_forecast_fallback_province(req.province),
            as_of_month=FIXED_AS_OF_DATE,
        )
        await _emit_stage("parse_intent", "我已根据你的描述解析出以下预测参数：", intent)
        _persist_checkpoint(FlowStage.PARSE_INTENT, payload={"intent": intent})
        cont_intent = await _wait_for_continue("confirm_intent", {"intent": intent})
        _persist_checkpoint(FlowStage.CONFIRM_INTENT, payload={"intent": intent, "continue": cont_intent})
        if cont_intent.get("approved") is False:
            raise RuntimeError("用户在参数确认阶段取消了任务")
        intent_overrides = normalize_continue_overrides(
            "confirm_intent",
            cont_intent.get("overrides") if isinstance(cont_intent.get("overrides"), dict) else {},
        )
        if intent_overrides:
            intent = apply_intent_overrides(intent, intent_overrides)
            await _emit_stage("intent_applied", "好的，我已按你的要求更新了预测参数：", intent)
            _persist_checkpoint(FlowStage.INTENT_APPLIED, payload={"intent": intent})

        out_root = _ensure_out_root(str(intent.get("province") or _forecast_fallback_province(req.province)))

        province_name = str(intent.get("province") or _forecast_fallback_province(req.province))
        from quantaalpha.forecast_agent.data import find_processed_excel

        data_path = await asyncio.to_thread(find_processed_excel, province_name)
        data_path_display = _display_path_for_ui(data_path)
        province_label = _province_display_label(province_name)
        await _emit_stage(
            "data_loading",
            f"正在导入{province_label}历史数据（{data_path_display}）",
            {"province": province_name, "dataPath": data_path_display},
        )
        _persist_checkpoint(
            FlowStage.DATA_LOADING,
            payload={"province": province_name, "dataPath": data_path_display},
        )

        # --- 阶段 1：xgboost 全特征诊断（与用户最终特征无关，只产出 importance）---
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("analyzing", 30, "运行 xgboost 诊断并提取重要性..."), "timestamp": _now()})
        diagnose = await _run_tool(
            FlowStage.DIAGNOSE,
            "diagnose_importance",
            intent,
            input_summary={"intent": intent, "contextLen": int(req.contextLen or 270)},
            output_dir=out_root,
            context_len=int(req.contextLen or 270),
        )
        await _emit_stage(
            "diagnose",
            "我已完成 xgboost 特征重要性诊断，以下是 Top 特征：",
            {
                "topFeatures": diagnose.get("importance_items", [])[: int(req.importanceTopK or 12)],
                "testMetrics": diagnose.get("test_metrics", {}),
            },
        )
        _persist_checkpoint(
            FlowStage.DIAGNOSE,
            payload={
                "topFeatures": diagnose.get("importance_items", [])[: int(req.importanceTopK or 12)],
                "testMetrics": diagnose.get("test_metrics", {}),
            },
        )

        # --- 阶段 2：LLM 特征推荐 + 确认点②（特征）---
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("planning", 50, "LLM 推荐特征组合..."), "timestamp": _now()})
        recommend = await _run_tool(
            FlowStage.RECOMMEND,
            "recommend_feature_superset",
            input_summary={
                "query": req.query,
                "intent": intent,
                "maxFeatureCount": int(req.maxFeatureCount or 10),
                "importanceTopK": int(req.importanceTopK or 12),
            },
            query=req.query,
            intent=intent,
            importance_items=diagnose.get("importance_items", []),
            max_feature_count=int(req.maxFeatureCount or 10),
            top_k=int(req.importanceTopK or 12),
            required_features=list(req.requiredFeatures or []),
        )
        await _emit_stage(
            "recommend",
            "结合重要性证据，我为你推荐了以下特征组合：",
            {
                "featureSuperset": recommend.get("feature_superset", []),
                "reason": recommend.get("reason", ""),
            },
        )
        _persist_checkpoint(
            FlowStage.RECOMMEND,
            payload={
                "featureSuperset": recommend.get("feature_superset", []),
                "reason": recommend.get("reason", ""),
            },
        )
        cont_features = await _wait_for_continue(
            "confirm_features",
            {
                "featureSuperset": recommend.get("feature_superset", []),
                "reason": recommend.get("reason", ""),
            },
        )
        _persist_checkpoint(
            FlowStage.CONFIRM_FEATURES,
            payload={
                "featureSuperset": recommend.get("feature_superset", []),
                "continue": cont_features,
            },
        )
        if cont_features.get("approved") is False:
            raise RuntimeError("用户在特征确认阶段取消了任务")
        feature_overrides = normalize_continue_overrides(
            "confirm_features",
            cont_features.get("overrides") if isinstance(cont_features.get("overrides"), dict) else {},
        )
        fs = feature_overrides.get("featureSuperset")
        if isinstance(fs, list) and fs:
            recommend["feature_superset"] = fs
            await _emit_stage(
                "features_applied",
                "好的，我已按你的要求更新了特征组合：",
                {
                    "featureSuperset": recommend["feature_superset"],
                    "reason": recommend.get("reason", ""),
                },
            )
            _persist_checkpoint(
                FlowStage.FEATURES_APPLIED,
                payload={"featureSuperset": recommend.get("feature_superset", [])},
            )

        # --- 阶段 3：多模型比选 + 月度汇总（无第三个人机确认点）---
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("forecasting", 75, "执行多模型比选与预测..."), "timestamp": _now()})
        compare = await _run_tool(
            FlowStage.COMPARE,
            "run_compare_and_rollup",
            input_summary={
                "intent": intent,
                "featureSuperset": list(recommend.get("feature_superset") or []),
                "candidateModels": (req.candidateModels or ""),
            },
            intent=intent,
            feature_superset=list(recommend.get("feature_superset") or []),
            output_dir=out_root,
            context_len=int(req.contextLen or 270),
            candidate_models=(req.candidateModels or ""),
        )
        selected_model = str(compare.get("best_model") or "")
        output_display = _display_path_for_ui(str(out_root))
        forecast_curve: list[dict[str, Any]] = []
        if selected_model:
            forecast_curve = _load_forecast_curve_for_ui(out_root / selected_model, selected_model)
        compare_payload = {
            "bestModel": compare.get("best_model"),
            "leaderboard": compare.get("leaderboard", []),
            "monthlyRollup": compare.get("monthly_rollup", []),
            "outputDir": output_display,
            "forecastCurve": forecast_curve,
        }
        await _emit_stage(
            "compare",
            f"多模型比选已完成，结果已存入 {output_display}：",
            compare_payload,
        )
        _persist_checkpoint(FlowStage.COMPARE, payload=compare_payload)

        payload: Dict[str, Any] = {
            "query": req.query,
            "intent": intent,
            "diagnose": {
                "summary_json": diagnose.get("summary_json"),
                "feature_importance_top": diagnose.get("importance_items", [])[: int(req.importanceTopK or 12)],
                "test_metrics": diagnose.get("test_metrics", {}),
            },
            "recommend": {
                "feature_superset": recommend.get("feature_superset", []),
                "reason": recommend.get("reason", ""),
            },
            "compare": {
                **compare,
                "output_dir_display": output_display,
                "forecast_curve": forecast_curve,
            },
            "selected_model": selected_model,
            "output_dir": str(out_root),
        }

        # --- 阶段 4（可选）：启动时预填 qaQuery 则自动问答一次；完成后多轮追问走 POST /forecast/qa ---
        if req.qaQuery and str(req.qaQuery).strip() and selected_model:
            await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("qa", 90, "生成问答结果..."), "timestamp": _now()})
            qa = await _run_tool(
                FlowStage.QA,
                "ask_flow_qa",
                input_summary={"model": selected_model, "query": str(req.qaQuery)},
                output_dir=out_root,
                model=selected_model,
                query=str(req.qaQuery),
                selected_features=list(recommend.get("feature_superset") or []),
            )
            payload["qa"] = qa
            await _emit_stage("qa", "已完成结果问答。", qa)
            _persist_checkpoint(FlowStage.QA, payload=qa)

        result_json = out_root / "agent_flow_result.json"
        result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        task["metrics"] = payload
        task["status"] = "completed"
        task["updatedAt"] = _now()
        task["resumeFromStage"] = None
        task["resumeState"] = None
        _persist_checkpoint(
            FlowStage.COMPLETED,
            payload={
                "selected_model": selected_model,
                "output_dir": str(out_root) if out_root is not None else "",
            },
            status="completed",
        )
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("completed", 100, "预测智能体流程完成"), "timestamp": _now()})
        await _broadcast(
            task_id,
            {
                "type": "result",
                "taskId": task_id,
                "data": {"status": "completed", "metrics": payload},
                "timestamp": _now(),
            },
        )
    except asyncio.CancelledError:
        task["status"] = "cancelled"
        task["updatedAt"] = _now()
        task["progress"]["message"] = "任务已取消"
        _persist_checkpoint(
            FlowStage.FAILED,
            payload={"message": "任务已取消"},
            status="failed",
            error="cancelled",
        )
        await _broadcast(
            task_id,
            {
                "type": "result",
                "taskId": task_id,
                "data": {"status": "cancelled"},
                "timestamp": _now(),
            },
        )
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["progress"]["message"] = str(e)
        task["updatedAt"] = _now()
        _persist_checkpoint(
            FlowStage.FAILED,
            payload={"message": str(e)},
            status="failed",
            error=str(e),
        )
        await _broadcast(
            task_id,
            {
                "type": "error",
                "taskId": task_id,
                "data": {"error": str(e)},
                "timestamp": _now(),
            },
        )


@app.post("/api/v1/forecast/agent/start", response_model=ApiResponse)
async def start_forecast_agent(req: ForecastAgentStartRequest):
    """Start single-round forecast agent flow task."""
    task_id = _gen_id()
    task = {
        "taskId": task_id,
        "status": "running",
        "type": "forecast_agent",
        "config": {**req.model_dump()},
        "progress": {
            "phase": "parsing",
            "currentRound": 0,
            "totalRounds": 1,
            "progress": 0,
            "message": "正在启动预测智能体...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {},
        "result": None,
        "pid": None,
        "flowState": FlowState(task_id=task_id).to_dict(),
        "lastCheckpoint": None,
        "checkpointPath": None,
        "checkpointPathDisplay": None,
        "auditPath": None,
        "auditPathDisplay": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task
    _spawn_forecast_runtime(task_id, req)
    return ApiResponse(
        success=True,
        data={"taskId": task_id, "task": _task_for_api(task)},
        message="预测智能体已启动",
    )


@app.get("/api/v1/forecast/agent/{task_id}", response_model=ApiResponse)
async def get_forecast_agent_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": _task_for_api(tasks[task_id])})


@app.delete("/api/v1/forecast/agent/{task_id}", response_model=ApiResponse)
async def cancel_forecast_agent(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    _forecast_continue_payloads.pop(task_id, None)
    cont_event = _forecast_continue_events.pop(task_id, None)
    if isinstance(cont_event, asyncio.Event):
        cont_event.set()
    runtime_task = _forecast_runtime_tasks.get(task_id)
    if isinstance(runtime_task, asyncio.Task) and not runtime_task.done():
        runtime_task.cancel()
    await _broadcast(task_id, {
        "type": "result",
        "taskId": task_id,
        "data": {"status": "cancelled"},
        "timestamp": _now(),
    })
    return ApiResponse(success=True, message="预测智能体任务已取消")


@app.post("/api/v1/forecast/agent/{task_id}/continue", response_model=ApiResponse)
async def continue_forecast_agent(task_id: str, req: ForecastAgentContinueRequest):
    """用户在人机确认点的回复入口。

    调用 gas_forecast_flow.parse_continue_message（仅 LLM 解析 approved/overrides），
    写入 _forecast_continue_payloads 并唤醒 _wait_for_continue 中阻塞的协程。
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    awaiting = ((task.get("metrics") or {}).get("awaiting_confirmation") or {})
    if not awaiting:
        raise HTTPException(status_code=409, detail="当前任务不在等待确认状态")
    if req.checkpoint and req.checkpoint != awaiting.get("checkpoint"):
        raise HTTPException(
            status_code=409,
            detail=f"checkpoint 不匹配，当前等待: {awaiting.get('checkpoint')}",
        )
    user_message = str(req.message or "").strip()
    checkpoint = str(awaiting.get("checkpoint") or "")
    normalized_req_overrides = req.overrides if isinstance(req.overrides, dict) else {}
    if not user_message and not normalized_req_overrides:
        raise HTTPException(status_code=400, detail="继续确认内容不能为空，请输入回复或传入 overrides")
    from quantaalpha.forecast_agent.gas_forecast_flow import parse_continue_message

    try:
        parsed_continue = parse_continue_message(
            checkpoint,
            awaiting.get("payload") or {},
            user_message,
            req_overrides=normalized_req_overrides,
            req_approved=bool(req.approved),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    approved = bool(parsed_continue.get("approved", True))
    overrides = parsed_continue.get("overrides") if isinstance(parsed_continue.get("overrides"), dict) else {}

    _forecast_continue_payloads[task_id] = {
        "approved": approved,
        "overrides": overrides,
        "message": user_message,
    }
    event = _forecast_continue_events.get(task_id)
    if isinstance(event, asyncio.Event):
        event.set()

    if user_message:
        if not isinstance(task.get("metrics"), dict):
            task["metrics"] = {}
        msg_list = task["metrics"].get("agent_messages")
        if not isinstance(msg_list, list):
            msg_list = []
        user_msg = {
            "role": "user",
            "stage": awaiting.get("checkpoint") or "",
            "messageType": "user_reply",
            "text": user_message,
            "payload": {"approved": approved, "overrides": overrides},
            "needConfirm": False,
            "timestamp": _now(),
        }
        msg_list.append(user_msg)
        task["metrics"]["agent_messages"] = msg_list[-200:]
        task["metrics"]["agent_message"] = user_msg
        await _broadcast(
            task_id,
            {
                "type": "metrics",
                "taskId": task_id,
                "data": {"agent_message": user_msg},
                "timestamp": _now(),
            },
        )

    log_entry = {
        "id": _gen_id(),
        "timestamp": _now(),
        "level": "info",
        "message": f"收到继续确认: checkpoint={awaiting.get('checkpoint')} approved={approved} overrides={json.dumps(overrides, ensure_ascii=False)}",
    }
    task["logs"].append(log_entry)
    if len(task["logs"]) > 2000:
        task["logs"] = task["logs"][-2000:]
    await _broadcast(task_id, {
        "type": "log",
        "taskId": task_id,
        "data": log_entry,
        "timestamp": _now(),
    })
    return ApiResponse(success=True, data={"task": _task_for_api(task)}, message="已提交继续指令")


def _load_forecast_test_points_for_ui(model_dir: Path, model_name: str) -> list[dict[str, Any]]:
    """读取测试集预测点（日期、预测值、真实值），供曲线与表格共用。"""
    import pandas as pd

    from quantaalpha.forecast_agent.data import normalize_period, period_to_start_date

    test_xlsx = model_dir / f"{model_name}_test.xlsx"
    test_csv = model_dir / f"{model_name}_test.csv"
    if test_xlsx.exists():
        test_df = pd.read_excel(test_xlsx)
    elif test_csv.exists():
        test_df = pd.read_csv(test_csv)
    else:
        test_df = None

    if test_df is None or not {"predicted_gas_sales", "actual_gas_sales"}.issubset(set(test_df.columns)):
        return []

    df = test_df.copy()
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower() == "test"].copy()
    if df.empty:
        return []

    def _row_ds(row: pd.Series) -> str | None:
        if "date" in df.columns and pd.notna(row.get("date")):
            try:
                return pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            except Exception:
                return None
        if "month" in df.columns and pd.notna(row.get("month")):
            try:
                return period_to_start_date(normalize_period(row["month"])).strftime("%Y-%m-%d")
            except Exception:
                return None
        return None

    points: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ds = _row_ds(row)
        if not ds:
            continue
        yhat = row.get("predicted_gas_sales")
        y = row.get("actual_gas_sales")
        if pd.isna(yhat) and pd.isna(y):
            continue
        period = str(row.get("month", "")) if "month" in df.columns and pd.notna(row.get("month")) else ""
        point: dict[str, Any] = {"ds": ds, "period": period}
        if pd.notna(yhat):
            point["yhat"] = float(yhat)
        if pd.notna(y):
            point["y"] = float(y)
        if pd.notna(yhat) and pd.notna(y) and float(y) != 0:
            point["error_pct"] = (float(yhat) - float(y)) / float(y) * 100.0
        points.append(point)

    points.sort(key=lambda p: p.get("ds", ""))
    return points


def _load_forecast_curve_for_ui(model_dir: Path, model_name: str) -> list[dict[str, Any]]:
    """读取测试集曲线（优先），供前端绘制预测值 vs 真实值。"""
    points = _load_forecast_test_points_for_ui(model_dir, model_name)
    if points:
        curve: list[dict[str, Any]] = []
        for p in points:
            item: dict[str, Any] = {"ds": p["ds"]}
            if p.get("period"):
                item["period"] = p["period"]
            if p.get("yhat") is not None:
                item["yhat"] = p["yhat"]
            if p.get("y") is not None:
                item["y"] = p["y"]
            curve.append(item)
        return curve

    # Fallback: forecast csv (prediction only).
    import pandas as pd

    csv_path = model_dir / f"{model_name}_forecast.csv"
    if not csv_path.exists():
        return []
    forecast_df = pd.read_csv(csv_path)
    if "ds" not in forecast_df.columns or "yhat" not in forecast_df.columns:
        return []
    return [{"ds": str(r["ds"])[:10], "yhat": float(r["yhat"])} for r in forecast_df.to_dict(orient="records")]


# ---- System config endpoints ----

@app.get("/api/v1/system/config", response_model=ApiResponse)
async def get_system_config():
    """Read current system configuration from .env."""
    dotenv = _load_dotenv_dict()

    # Mask API keys for security
    masked_env = {}
    for k, v in dotenv.items():
        if "KEY" in k.upper() and v:
            masked_env[k] = v[:8] + "..." + v[-4:] if len(v) > 12 else "***"
        else:
            masked_env[k] = v

    return ApiResponse(
        success=True,
        data={
            "env": masked_env,
            "forecastConfig": str(PROJECT_ROOT / "configs" / "forecast.yaml"),
        },
    )


@app.put("/api/v1/system/config", response_model=ApiResponse)
async def update_system_config(update: SystemConfigUpdate):
    """Update .env configuration (non-secret fields only)."""
    if not DOTENV_PATH.exists():
        raise HTTPException(status_code=404, detail=".env file not found")

    content = DOTENV_PATH.read_text(encoding="utf-8")
    updates = {k: v for k, v in update.model_dump().items() if v is not None}

    import re
    for key, val in updates.items():
        # Replace existing line or append
        pattern = rf"^{re.escape(key)}\s*=.*$"
        replacement = f"{key}={val}"
        new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if n > 0:
            content = new_content
        else:
            content += f"\n{replacement}\n"

    DOTENV_PATH.write_text(content, encoding="utf-8")
    return ApiResponse(success=True, message="配置已更新")


# ---- WebSocket endpoint ----

@app.websocket("/ws/mining/{task_id}")
async def ws_mining(websocket: WebSocket, task_id: str):
    """WebSocket for real-time experiment updates."""
    await websocket.accept()

    if task_id not in ws_connections:
        ws_connections[task_id] = []
    ws_connections[task_id].append(websocket)

    # Send current state immediately
    if task_id in tasks:
        try:
            await websocket.send_json({
                "type": "progress",
                "taskId": task_id,
                "data": tasks[task_id].get("progress", {}),
                "timestamp": _now(),
            })
            # Send recent logs
            for log in tasks[task_id].get("logs", [])[-20:]:
                await websocket.send_json({
                    "type": "log",
                    "taskId": task_id,
                    "data": log,
                    "timestamp": _now(),
                })
        except Exception:
            pass

    try:
        while True:
            data = await websocket.receive_text()
            # Heartbeat
            if data == "ping":
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": _now(),
                })
    except WebSocketDisconnect:
        if task_id in ws_connections:
            try:
                ws_connections[task_id].remove(websocket)
            except ValueError:
                pass


# ========================== Entry Point ==========================

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")

"""
QuantaAlpha Backend API
FastAPI-based REST + WebSocket API for factor mining and backtesting.

Integrates with the core QuantaAlpha CLI to launch experiments
and reads factor library JSON for the factor browsing API.
"""

import asyncio
import glob
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
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

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
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

# ========================== Pydantic Models ==========================


class MiningStartRequest(BaseModel):
    """Request to start a factor mining experiment."""
    direction: str = Field(..., description="Research direction, e.g. '价量因子挖掘'")
    numDirections: Optional[int] = Field(2, description="Parallel exploration directions")
    maxRounds: Optional[int] = Field(3, description="Evolution rounds")
    maxLoops: Optional[int] = Field(2, description="Iterations per direction")
    factorsPerHypothesis: Optional[int] = Field(3, description="Factors per hypothesis")
    librarySuffix: Optional[str] = Field(None, description="Factor library file suffix")
    qualityGateEnabled: Optional[bool] = Field(None, description="Enable quality gate checks")
    parallelEnabled: Optional[bool] = Field(None, description="Enable parallel execution within evolution phases")


class BacktestStartRequest(BaseModel):
    """Request to start an independent backtest."""
    factorJson: str = Field(..., description="Path to factor library JSON")
    factorSource: str = Field("custom", description="custom | combined")
    configPath: Optional[str] = Field(None, description="Path to backtest config")


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


# ========================== Utility Helpers ==========================

def _gen_id() -> str:
    return str(uuid.uuid4())[:8]


def _now() -> str:
    return datetime.now().isoformat()


def _task_for_api(task: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable copy of task state (strip runtime-only fields)."""
    return {k: v for k, v in task.items() if not str(k).startswith("_")}

def _decode_subprocess_line(line_bytes: bytes) -> str:
    """Best-effort decode for Windows consoles and UTF-8 Python output."""
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return line_bytes.decode(enc)
        except Exception:
            continue
    return line_bytes.decode("utf-8", errors="replace")


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


def _find_factor_jsons() -> List[str]:
    """Find all factor library JSON files in data/factorlib/."""
    factorlib_dir = PROJECT_ROOT / "data" / "factorlib"
    pattern = str(factorlib_dir / "all_factors_library*.json")
    results = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    old_pattern = str(PROJECT_ROOT / "all_factors_library*.json")
    old_results = sorted(glob.glob(old_pattern), key=os.path.getmtime, reverse=True)

    seen = set(results)
    for r in old_results:
        if r not in seen:
            results.append(r)
    return results


def _load_factor_library(path: str) -> Dict[str, Any]:
    """Load and parse a factor library JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _classify_quality(backtest_results: Dict[str, Any]) -> str:
    """Classify factor quality based on backtest metrics."""
    if not backtest_results:
        return "low"
    # Use information ratio or IC-related metrics
    ic = None
    for key in ["1day.excess_return_without_cost.information_ratio",
                 "1day.excess_return_with_cost.information_ratio"]:
        if key in backtest_results:
            ic = backtest_results[key]
            break
    if ic is None:
        # Try to find any IC-like metric
        for key, val in backtest_results.items():
            if "information_ratio" in key and isinstance(val, (int, float)):
                ic = val
                break
    if ic is None:
        return "medium"
    if ic > 0.5:
        return "high"
    if ic > 0.1:
        return "medium"
    return "low"


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


# ========================== Mining Process ==========================

async def _run_mining(task_id: str, req: MiningStartRequest):
    """
    Launch the actual QuantaAlpha mining experiment as a subprocess
    and stream its output over WebSocket.
    """
    task = tasks[task_id]
    try:
        # Build the command
        env = os.environ.copy()
        # Load .env into env
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # Use experiment_id as suffix to guarantee isolation
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        env["EXPERIMENT_ID"] = experiment_id
        
        # Enforce unique library suffix if not provided
        if not req.librarySuffix:
            req.librarySuffix = experiment_id
            # Update task config so frontend knows the suffix
            task["config"]["librarySuffix"] = req.librarySuffix
            
        env["FACTOR_LIBRARY_SUFFIX"] = req.librarySuffix

        results_base = dotenv.get("DATA_RESULTS_DIR", str(PROJECT_ROOT / "data" / "results"))
        env["WORKSPACE_PATH"] = f"{results_base}/workspace_{experiment_id}"
        env["PICKLE_CACHE_FOLDER_PATH_STR"] = f"{results_base}/pickle_cache_{experiment_id}"

        os.makedirs(env["WORKSPACE_PATH"], exist_ok=True)
        os.makedirs(env["PICKLE_CACHE_FOLDER_PATH_STR"], exist_ok=True)

        # Qlib symlink
        qlib_data = dotenv.get("QLIB_DATA_DIR", "")
        if qlib_data:
            qlib_symlink_dir = Path.home() / ".qlib" / "qlib_data"
            qlib_symlink_dir.mkdir(parents=True, exist_ok=True)
            cn_data_link = qlib_symlink_dir / "cn_data"
            if not cn_data_link.exists() or os.readlink(str(cn_data_link)) != qlib_data:
                if cn_data_link.is_symlink():
                    cn_data_link.unlink()
                cn_data_link.symlink_to(qlib_data)

        # Build a temporary config with frontend parameter overrides
        base_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
        config_path_to_use = str(base_config_path)

        try:
            with open(base_config_path, "r", encoding="utf-8") as _f:
                run_cfg = yaml.safe_load(_f) or {}

            # Apply frontend overrides
            if req.numDirections is not None:
                run_cfg.setdefault("planning", {})["num_directions"] = req.numDirections
            if req.maxRounds is not None:
                run_cfg.setdefault("evolution", {})["max_rounds"] = req.maxRounds
            if req.maxLoops is not None:
                run_cfg.setdefault("execution", {})["max_loops"] = req.maxLoops
            if req.factorsPerHypothesis is not None:
                run_cfg.setdefault("factor", {})["factors_per_hypothesis"] = req.factorsPerHypothesis

            # Apply parallel execution override from frontend
            if req.parallelEnabled is not None:
                run_cfg.setdefault("evolution", {})["parallel_enabled"] = req.parallelEnabled
                run_cfg.setdefault("execution", {})["parallel_execution"] = req.parallelEnabled

            # Apply quality gate override from frontend
            if req.qualityGateEnabled is not None:
                qg = run_cfg.setdefault("quality_gate", {})
                if req.qualityGateEnabled:
                    # Enable quality gate: enable complexity and redundancy checks (default on), consistency keeps user YAML setting
                    qg.setdefault("complexity_enabled", True)
                    qg.setdefault("redundancy_enabled", True)
                    # Consistency check is expensive, only enable if explicitly enabled in YAML
                    qg.setdefault("consistency_enabled", False)
                else:
                    # Disable quality gate: disable all
                    qg["consistency_enabled"] = False
                    qg["complexity_enabled"] = False
                    qg["redundancy_enabled"] = False

            # Write to a temporary file so the original is untouched
            tmp_dir = Path(env.get("WORKSPACE_PATH", "/tmp"))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_cfg = tmp_dir / "experiment_override.yaml"
            with open(tmp_cfg, "w", encoding="utf-8") as _f:
                yaml.safe_dump(run_cfg, _f, allow_unicode=True, default_flow_style=False)
            config_path_to_use = str(tmp_cfg)
        except Exception as cfg_err:
            # Fall back to original config if anything fails
            import traceback
            traceback.print_exc()

        # Build CLI args
        cmd = [
            sys.executable, "-m", "quantaalpha.cli", "mine",
            "--direction", req.direction,
            "--config_path", config_path_to_use,
        ]

        task["status"] = "running"
        task["progress"]["phase"] = "planning"
        task["progress"]["message"] = "正在启动实验..."
        task["updatedAt"] = _now()

        await _broadcast(task_id, {
            "type": "progress",
            "taskId": task_id,
            "data": task["progress"],
            "timestamp": _now(),
        })

        # Launch subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        task["pid"] = proc.pid

        # Stream stdout line by line
        line_count = 0
        current_phase = "planning"

        # Noisy patterns to suppress (shared with backtest)
        _MINING_NOISE = (
            "field data contains nan",
            "common_infra",
            "PyTorch models are skipped",
            "UserWarning: pkg_resources",
            "FutureWarning",
            "UserWarning",
            "Training until validation scores",
            "Did not meet early stopping",
        )

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            line_count += 1

            # Skip noisy warnings
            if any(p in line for p in _MINING_NOISE):
                continue

            # Detect phase from log messages
            new_phase = current_phase
            if "factor_propose" in line:
                new_phase = "evolving"
            elif "factor_backtest" in line or "backtest" in line.lower():
                new_phase = "backtesting"
            elif "feedback" in line:
                new_phase = "analyzing"
            elif "factor_calculate" in line:
                new_phase = "evolving"
            elif "规划" in line or "planning" in line.lower():
                new_phase = "planning"
            elif "进化完成" in line or "程序执行完成" in line:
                new_phase = "completed"

            if new_phase != current_phase:
                current_phase = new_phase
                task["progress"]["phase"] = current_phase
                task["progress"]["message"] = line[:200]
                task["progress"]["timestamp"] = _now()
                await _broadcast(task_id, {
                    "type": "progress",
                    "taskId": task_id,
                    "data": task["progress"],
                    "timestamp": _now(),
                })

            # Send log every line (throttle to avoid flooding)
            if line_count % 3 == 0 or "INFO" in line or "ERROR" in line or "WARNING" in line:
                level = "info"
                if "ERROR" in line or "Error" in line:
                    level = "error"
                elif "WARNING" in line or "Warning" in line:
                    level = "warning"
                elif "完成" in line or "success" in line.lower():
                    level = "success"

                log_entry = {
                    "id": _gen_id(),
                    "timestamp": _now(),
                    "level": level,
                    "message": line[:500],
                }
                task["logs"].append(log_entry)
                # Keep only last 500 logs in memory
                if len(task["logs"]) > 500:
                    task["logs"] = task["logs"][-500:]

                await _broadcast(task_id, {
                    "type": "log",
                    "taskId": task_id,
                    "data": log_entry,
                    "timestamp": _now(),
                })

            # Extract metrics from log lines like "RankIC=0.0016"
            if "RankIC=" in line:
                try:
                    rank_ic_str = line.split("RankIC=")[1].split(",")[0].split(")")[0]
                    task["metrics"]["rankIc"] = float(rank_ic_str)
                    await _broadcast(task_id, {
                        "type": "metrics",
                        "taskId": task_id,
                        "data": task["metrics"],
                        "timestamp": _now(),
                    })
                except Exception:
                    pass
            
            # Check for factor saving to update top factors list
            if "已保存" in line or "因子" in line:
                _update_mining_metrics(task)
                if task.get("metrics"):
                     await _broadcast(task_id, {
                        "type": "result",
                        "taskId": task_id,
                        "data": {"status": task["status"], "metrics": task["metrics"]},
                        "timestamp": _now(),
                    })

        exit_code = await proc.wait()
        task["pid"] = None

        if exit_code == 0:
            task["status"] = "completed"
            task["progress"]["phase"] = "completed"
            task["progress"]["progress"] = 100
            task["progress"]["message"] = "实验完成"
        else:
            task["status"] = "failed"
            task["progress"]["message"] = f"实验失败 (exit code: {exit_code})"

        task["updatedAt"] = _now()

        # Load final factor count from the library JSON
        # Prefer the library file matching the librarySuffix for this experiment
        _update_mining_metrics(task)

        await _broadcast(task_id, {
            "type": "result",
            "taskId": task_id,
            "data": {"status": task["status"], "metrics": task["metrics"]},
            "timestamp": _now(),
        })

    except Exception as e:
        task["status"] = "failed"
        task["progress"]["message"] = f"Error: {str(e)}"
        task["updatedAt"] = _now()
        await _broadcast(task_id, {
            "type": "error",
            "taskId": task_id,
            "data": {"error": str(e)},
            "timestamp": _now(),
        })


# ========================== API Endpoints ==========================

@app.get("/")
async def root():
    return {"message": "QuantaAlpha API", "version": "2.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": _now()}


# ---- Mining endpoints ----

@app.post("/api/v1/mining/start", response_model=ApiResponse)
async def start_mining(req: MiningStartRequest):
    """Start a new factor mining experiment."""
    task_id = _gen_id()
    task = {
        "taskId": task_id,
        "status": "running",
        "config": req.model_dump(),
        "progress": {
            "phase": "parsing",
            "currentRound": 0,
            "totalRounds": req.maxRounds or 3,
            "progress": 0,
            "message": "正在初始化实验...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {
            "ic": 0, "icir": 0, "rankIc": 0, "rankIcir": 0,
            "annualReturn": 0, "sharpeRatio": 0, "maxDrawdown": 0,
            "totalFactors": 0, "highQualityFactors": 0,
            "mediumQualityFactors": 0, "lowQualityFactors": 0,
        },
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task

    # Launch the mining process in background
    asyncio.create_task(_run_mining(task_id, req))

    return ApiResponse(
        success=True,
        data={"taskId": task_id, "task": task},
        message="实验已启动",
    )


@app.get("/api/v1/mining/{task_id}", response_model=ApiResponse)
async def get_mining_status(task_id: str):
    """Get task status."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": tasks[task_id]})


@app.delete("/api/v1/mining/{task_id}", response_model=ApiResponse)
async def cancel_mining(task_id: str):
    """Cancel a running mining task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("pid"):
        try:
            pid = task["pid"]
            # Try graceful termination first
            os.kill(pid, signal.SIGTERM)
            
            # Wait briefly for cleanup (0.5s)
            for _ in range(5):
                try:
                    os.kill(pid, 0) # Check if alive
                    await asyncio.sleep(0.1)
                except ProcessLookupError:
                    break
            
            # Force kill if still running
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    await _broadcast(task_id, {
        "type": "result",
        "taskId": task_id,
        "data": {"status": "cancelled"},
        "timestamp": _now(),
    })
    return ApiResponse(success=True, message="任务已取消")


@app.get("/api/v1/mining/tasks/list", response_model=ApiResponse)
async def list_tasks():
    """List all tasks."""
    task_list = sorted(tasks.values(), key=lambda t: t["createdAt"], reverse=True)
    return ApiResponse(success=True, data={"tasks": task_list})


# ---- Factor library endpoints ----

@app.get("/api/v1/factors", response_model=ApiResponse)
async def get_factors(
    quality: Optional[str] = Query(None, description="Filter by quality: high/medium/low"),
    search: Optional[str] = Query(None, description="Search by factor name"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    library: Optional[str] = Query(None, description="Specific library file name"),
):
    """Get factors from the factor library JSON."""
    # Find the most recent factor library
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        # Fallback: check if file exists at project root (legacy location)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(
                success=True,
                data={"factors": [], "total": 0, "limit": limit, "offset": offset,
                      "libraries": []},
            )
        lib_path = jsons[0]

    try:
        raw = _load_factor_library(lib_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read factor library: {e}")

    factors_dict = raw.get("factors", {})
    metadata = raw.get("metadata", {})

    # Convert dict to list with quality classification
    factors_list: List[Dict[str, Any]] = []
    for factor_id, factor_info in factors_dict.items():
        if not isinstance(factor_info, dict):
            continue
        bt = factor_info.get("backtest_results", {})
        q = _classify_quality(bt)
        # Extract metrics with proper fallbacks
        # Try specific keys first, then standard ones
        ic = bt.get("IC", bt.get("1day.excess_return_without_cost.information_coefficient", 0))
        icir = bt.get("ICIR", bt.get("1day.excess_return_without_cost.information_coefficient_ir", 0))
        rank_ic = bt.get("Rank IC", bt.get("rank_ic", bt.get("1day.excess_return_without_cost.rank_ic", 0)))
        rank_icir = bt.get("Rank ICIR", bt.get("rank_ic_ir", bt.get("1day.excess_return_without_cost.rank_ic_ir", 0)))
        
        factor_entry = {
            "factorId": factor_info.get("factor_id", factor_id),
            "factorName": factor_info.get("factor_name", "Unknown"),
            "factorExpression": factor_info.get("factor_expression", ""),
            "factorDescription": factor_info.get("factor_description", ""),
            "factorFormulation": factor_info.get("factor_formulation", ""),
            "quality": q,
            "backtestResults": bt,
            # Extract key metrics
            "ic": ic,
            "icir": icir,
            "rankIc": rank_ic,
            "rankIcir": rank_icir,
            "annualReturn": bt.get("1day.excess_return_with_cost.annualized_return", 
                                  bt.get("1day.excess_return_without_cost.annualized_return", 0)),
            "maxDrawdown": bt.get("1day.excess_return_with_cost.max_drawdown", 
                                 bt.get("1day.excess_return_without_cost.max_drawdown", 0)),
            "sharpeRatio": bt.get("1day.excess_return_with_cost.information_ratio", 
                                bt.get("1day.excess_return_without_cost.information_ratio", 0)),
            "round": factor_info.get("evolution_metadata", {}).get("round", 0)
            if isinstance(factor_info.get("evolution_metadata"), dict) else 0,
            "direction": factor_info.get("evolution_metadata", {}).get("direction_index", "")
            if isinstance(factor_info.get("evolution_metadata"), dict) else "",
            "createdAt": factor_info.get("added_at", ""),
        }
        factors_list.append(factor_entry)

    # Apply filters
    if quality:
        factors_list = [f for f in factors_list if f["quality"] == quality]
    if search:
        search_lower = search.lower()
        factors_list = [
            f for f in factors_list
            if search_lower in f["factorName"].lower()
            or search_lower in f.get("factorDescription", "").lower()
            or search_lower in f.get("factorExpression", "").lower()
        ]

    total = len(factors_list)
    paginated = factors_list[offset: offset + limit]

    # Available library files
    all_libs = [Path(p).name for p in _find_factor_jsons()]

    return ApiResponse(
        success=True,
        data={
            "factors": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "metadata": metadata,
            "libraries": all_libs,
        },
    )


# ---- Factor cache endpoints ----
# IMPORTANT: These must be registered BEFORE /api/v1/factors/{factor_id}
# otherwise FastAPI matches "cache-status" as a factor_id parameter.

@app.get("/api/v1/factors/cache-status", response_model=ApiResponse)
async def get_cache_status(
    library: Optional[str] = Query(None, description="Factor library JSON filename"),
):
    """Check cache status of factors in the specified factor library."""
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(success=True, data={
                "total": 0, "h5_cached": 0, "md5_cached": 0,
                "need_compute": 0, "factors": [],
            })
        lib_path = jsons[0]

    if not Path(lib_path).exists():
        raise HTTPException(status_code=404, detail=f"Factor library not found: {library}")

    # Import from core library
    from quantaalpha.factors.library import FactorLibraryManager
    result = FactorLibraryManager.check_cache_status(lib_path)
    return ApiResponse(success=True, data=result)


@app.post("/api/v1/factors/warm-cache", response_model=ApiResponse)
async def warm_cache(
    library: Optional[str] = Query(None, description="Factor library JSON filename"),
):
    """Batch sync from result.h5 to MD5 cache directory."""
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(success=False, error="未找到因子库文件")
        lib_path = jsons[0]

    if not Path(lib_path).exists():
        raise HTTPException(status_code=404, detail=f"Factor library not found: {library}")

    from quantaalpha.factors.library import FactorLibraryManager
    result = FactorLibraryManager.warm_cache_from_json(lib_path)
    # Build a clear message
    parts = []
    if result['synced']:
        parts.append(f"新同步 {result['synced']} 个")
    if result.get('already_cached'):
        parts.append(f"已有缓存 {result['already_cached']} 个")
    if result.get('no_source'):
        parts.append(f"无H5源 {result['no_source']} 个(回测时从表达式计算)")
    if result['failed']:
        parts.append(f"失败 {result['failed']} 个")
    msg = "，".join(parts) if parts else "无需操作"
    return ApiResponse(
        success=True,
        data=result,
        message=msg,
    )


# ---- Factor library list endpoint (must be BEFORE {factor_id} route) ----

@app.get("/api/v1/factors/libraries", response_model=ApiResponse)
async def list_factor_libraries():
    """List all factor library JSON files in the project root."""
    libs = [Path(p).name for p in _find_factor_jsons()]
    return ApiResponse(success=True, data={"libraries": libs})


@app.get("/api/v1/factors/{factor_id}", response_model=ApiResponse)
async def get_factor_detail(factor_id: str):
    """Get full detail of a single factor."""
    jsons = _find_factor_jsons()
    for lib_path in jsons:
        try:
            raw = _load_factor_library(lib_path)
            factors = raw.get("factors", {})
            if factor_id in factors:
                info = factors[factor_id]
                return ApiResponse(success=True, data={"factor": info})
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Factor not found")


# ---- Backtest endpoints ----

@app.post("/api/v1/backtest/start", response_model=ApiResponse)
async def start_backtest(req: BacktestStartRequest):
    """Start an independent backtest."""
    task_id = _gen_id()
    config_path = req.configPath or str(PROJECT_ROOT / "configs" / "backtest.yaml")

    task = {
        "taskId": task_id,
        "status": "running",
        "type": "backtest",
        "config": {**req.model_dump(), "configPath": config_path},
        "progress": {
            "phase": "backtesting",
            "currentRound": 0,
            "totalRounds": 1,
            "progress": 0,
            "message": "正在启动回测...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {},
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task

    # Launch backtest in background
    asyncio.create_task(_run_backtest(task_id, req, config_path))
    return ApiResponse(
        success=True,
        data={"taskId": task_id, "task": task},
        message="回测已启动",
    )


@app.get("/api/v1/backtest/{task_id}", response_model=ApiResponse)
async def get_backtest_status(task_id: str):
    """Get backtest task status and results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": tasks[task_id]})


@app.delete("/api/v1/backtest/{task_id}", response_model=ApiResponse)
async def cancel_backtest(task_id: str):
    """Cancel a running backtest task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("pid"):
        try:
            os.kill(task["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    await _broadcast(task_id, {
        "type": "result",
        "taskId": task_id,
        "data": {"status": "cancelled"},
        "timestamp": _now(),
    })
    return ApiResponse(success=True, message="回测已取消")


async def _run_backtest(task_id: str, req: BacktestStartRequest, config_path: str):
    """Run the independent backtest (V2) as a subprocess."""
    task = tasks[task_id]
    try:
        env = os.environ.copy()
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # --- Resolve factor JSON path ---
        # Frontend sends just the filename (e.g. "all_factors_library_test3hjback.json")
        # We need to resolve it to the full path under data/factorlib/
        factor_json_input = req.factorJson
        factor_json_path = Path(factor_json_input)
        if not factor_json_path.is_absolute():
            # Check data/factorlib/ first
            candidate = PROJECT_ROOT / "data" / "factorlib" / factor_json_input
            if candidate.exists():
                factor_json_path = candidate
            else:
                # Try as relative to project root
                candidate2 = PROJECT_ROOT / factor_json_input
                if candidate2.exists():
                    factor_json_path = candidate2
                else:
                    factor_json_path = candidate  # will fail with a clear error message
        factor_json_str = str(factor_json_path)

        # --- Find the correct Python executable ---
        # Prefer the conda env that has qlib installed
        conda_env = dotenv.get("CONDA_ENV_NAME", "quantaalpha")
        python_bin = sys.executable  # fallback

        # Dynamically detect conda base path (portable, no hardcoded paths)
        conda_prefixes = [os.path.expanduser(f"~/.conda/envs/{conda_env}")]
        try:
            import subprocess as _sp
            conda_base = _sp.check_output(
                ["conda", "info", "--base"], text=True, timeout=5
            ).strip()
            conda_prefixes.insert(0, os.path.join(conda_base, "envs", conda_env))
        except Exception:
            pass
        # Also check CONDA_PREFIX if we're already in the right env
        if os.environ.get("CONDA_PREFIX"):
            conda_prefixes.insert(0, os.environ["CONDA_PREFIX"])

        for prefix in conda_prefixes:
            candidate_bin = Path(prefix) / "bin" / "python"
            if candidate_bin.exists():
                python_bin = str(candidate_bin)
                break

        # Build CLI command
        cmd = [
            python_bin, "-m", "quantaalpha.backtest.run_backtest",
            "-c", config_path,
            "--factor-source", req.factorSource,
            "--factor-json", factor_json_str,
            "--skip-uncached",
            "-v",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        task["pid"] = proc.pid

        # Noisy warnings from Qlib / dependencies that can be safely suppressed
        _NOISY_PATTERNS = (
            "field data contains nan",
            "common_infra",
            "PyTorch models are skipped",
            "UserWarning: pkg_resources",
            "Training until validation scores",
            "FutureWarning",
            "UserWarning",
            "Did not meet early stopping",
            "num_leaves is set=",
        )

        # --- Stream stdout ---
        log_entry = None
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue

            # Skip noisy repeated warnings
            if any(p in line for p in _NOISY_PATTERNS):
                continue

            level = "info"
            if "ERROR" in line or "Error" in line:
                level = "error"
            elif "WARNING" in line or "Warning" in line:
                level = "warning"
            elif "完成" in line or "success" in line.lower() or "✓" in line:
                level = "success"

            log_entry = {
                "id": _gen_id(),
                "timestamp": _now(),
                "level": level,
                "message": line[:500],
            }
            task["logs"].append(log_entry)
            if len(task["logs"]) > 2000:
                task["logs"] = task["logs"][-2000:]

            # Broadcast log to WebSocket
            await _broadcast(task_id, {
                "type": "log",
                "taskId": task_id,
                "data": log_entry,
                "timestamp": _now(),
            })

            # Update progress for meaningful lines
            if any(kw in line for kw in ["因子", "回测", "模型", "训练", "完成", "加载",
                                          "[1/4]", "[2/4]", "[3/4]", "[4/4]", "结果"]):
                task["progress"]["message"] = line[:200]

                # Estimate progress from run_backtest step markers
                if "[1/4]" in line:
                    task["progress"]["progress"] = 15
                elif "[2/4]" in line:
                    task["progress"]["progress"] = 35
                elif "[3/4]" in line:
                    task["progress"]["progress"] = 55
                elif "[4/4]" in line:
                    task["progress"]["progress"] = 75
                elif "结果已保存" in line or "回测结果" in line:
                    task["progress"]["progress"] = 95

                task["progress"]["timestamp"] = _now()
                await _broadcast(task_id, {
                    "type": "progress",
                    "taskId": task_id,
                    "data": task["progress"],
                    "timestamp": _now(),
                })

        # --- Process exit ---
        exit_code = await proc.wait()
        task["pid"] = None
        task["status"] = "completed" if exit_code == 0 else "failed"
        task["updatedAt"] = _now()

        # Try to load backtest results from output metrics JSON
        if exit_code == 0:
            task["progress"]["phase"] = "completed"
            task["progress"]["progress"] = 100
            task["progress"]["message"] = "回测完成"
            _load_backtest_results(task)
        else:
            task["progress"]["message"] = f"回测失败 (exit code: {exit_code})"

        await _broadcast(task_id, {
            "type": "result",
            "taskId": task_id,
            "data": {
                "status": task["status"],
                "metrics": task.get("metrics", {}),
            },
            "timestamp": _now(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["progress"]["message"] = str(e)
        task["updatedAt"] = _now()
        await _broadcast(task_id, {
            "type": "error",
            "taskId": task_id,
            "data": {"error": str(e)},
            "timestamp": _now(),
        })


def _load_backtest_results(task: Dict[str, Any]):
    """Try to load backtest result metrics from the output directory."""
    try:
        config_path = task.get("config", {}).get("configPath") or str(
            PROJECT_ROOT / "configs" / "backtest.yaml"
        )
        with open(config_path, "r") as f:
            bt_config = yaml.safe_load(f)
        output_dir_raw = bt_config.get("experiment", {}).get(
            "output_dir", "data/results/backtest_v2_results"
        )
        # Resolve relative output_dir against PROJECT_ROOT (run_backtest runs with cwd=PROJECT_ROOT)
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir_str = str(output_dir)

        # Look for most recent metrics JSON
        metrics_files = sorted(
            glob.glob(os.path.join(output_dir_str, "*_backtest_metrics.json")),
            key=os.path.getmtime, reverse=True,
        )
        if metrics_files:
            with open(metrics_files[0], "r") as f:
                metrics_data = json.load(f)
            # The JSON has a nested structure: { metrics: {...}, config: {...}, ... }
            # Flatten: put the inner metrics dict at the top level for the frontend,
            # but also keep meta fields like experiment_name and elapsed_seconds.
            inner_metrics = metrics_data.get("metrics", {})
            flat = {**inner_metrics}
            # Carry over useful metadata
            for key in ("experiment_name", "factor_source", "num_factors",
                        "config", "elapsed_seconds"):
                if key in metrics_data:
                    flat[f"__{key}"] = metrics_data[key]
            
            # Load cumulative excess return data from CSV
            csv_path = metrics_files[0].replace("_backtest_metrics.json", "_cumulative_excess.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                if 'date' in df.columns and 'cumulative_excess_return' in df.columns:
                    cumulative_data = df[['date', 'cumulative_excess_return']].to_dict('records')
                    flat["cumulative_curve"] = [
                        {"date": r["date"], "value": r["cumulative_excess_return"]} 
                        for r in cumulative_data
                    ]

            task["metrics"] = flat
    except Exception as e:
        import traceback
        traceback.print_exc()  # print for debugging, but don't crash


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
            ask_flow_qa,
            diagnose_importance,
            normalize_continue_overrides,
            parse_intent,
            recommend_feature_superset,
            run_compare_and_rollup,
        )

        if not str(os.getenv("OPENAI_API_KEY", "")).strip():
            raise RuntimeError("缺少 OPENAI_API_KEY，请在项目根目录 .env 中配置后重试")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root: Path | None = None

        def _sync_task_output_config(intent_province: str, resolved_out_root: Path) -> None:
            if isinstance(task.get("config"), dict):
                task["config"]["province"] = intent_province
                task["config"]["outputDir"] = str(resolved_out_root)

        def _ensure_out_root(intent_province: str) -> Path:
            nonlocal out_root
            resolved = _resolve_agent_flow_out_root(
                intent_province,
                run_ts=ts,
                custom_output_dir=req.outputDir,
            )
            resolved.mkdir(parents=True, exist_ok=True)
            out_root = resolved
            _sync_task_output_config(intent_province, resolved)
            return resolved

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
        intent = await asyncio.to_thread(
            parse_intent,
            req.query,
            default_province=_forecast_fallback_province(req.province),
            as_of_month=FIXED_AS_OF_DATE,
        )
        await _emit_stage("parse_intent", "我已根据你的描述解析出以下预测参数：", intent)
        cont_intent = await _wait_for_continue("confirm_intent", {"intent": intent})
        if cont_intent.get("approved") is False:
            raise RuntimeError("用户在参数确认阶段取消了任务")
        intent_overrides = normalize_continue_overrides(
            "confirm_intent",
            cont_intent.get("overrides") if isinstance(cont_intent.get("overrides"), dict) else {},
        )
        if intent_overrides:
            intent = apply_intent_overrides(intent, intent_overrides)
            await _emit_stage("intent_applied", "好的，我已按你的要求更新了预测参数：", intent)

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

        # --- 阶段 1：xgboost 全特征诊断（与用户最终特征无关，只产出 importance）---
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("analyzing", 30, "运行 xgboost 诊断并提取重要性..."), "timestamp": _now()})
        diagnose = await asyncio.to_thread(
            diagnose_importance,
            intent,
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

        # --- 阶段 2：LLM 特征推荐 + 确认点②（特征）---
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("planning", 50, "LLM 推荐特征组合..."), "timestamp": _now()})
        recommend = await asyncio.to_thread(
            recommend_feature_superset,
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
        cont_features = await _wait_for_continue(
            "confirm_features",
            {
                "featureSuperset": recommend.get("feature_superset", []),
                "reason": recommend.get("reason", ""),
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

        # --- 阶段 3：多模型比选 + 月度汇总（无第三个人机确认点）---
        await _broadcast(task_id, {"type": "progress", "taskId": task_id, "data": _set_progress("forecasting", 75, "执行多模型比选与预测..."), "timestamp": _now()})
        compare = await asyncio.to_thread(
            run_compare_and_rollup,
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
            qa = await asyncio.to_thread(
                ask_flow_qa,
                output_dir=out_root,
                model=selected_model,
                query=str(req.qaQuery),
                selected_features=list(recommend.get("feature_superset") or []),
            )
            payload["qa"] = qa
            await _emit_stage("qa", "已完成结果问答。", qa)

        result_json = out_root / "agent_flow_result.json"
        result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        task["metrics"] = payload
        task["status"] = "completed"
        task["updatedAt"] = _now()
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["progress"]["message"] = str(e)
        task["updatedAt"] = _now()
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
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task
    asyncio.create_task(_run_forecast_agent(task_id, req))
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
    from quantaalpha.forecast_agent.gas_forecast_flow import parse_continue_message

    try:
        parsed_continue = parse_continue_message(
            checkpoint,
            awaiting.get("payload") or {},
            user_message,
            req_overrides=req.overrides,
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
    """Read current system configuration from .env and experiment.yaml."""
    dotenv = _load_dotenv_dict()

    # Read experiment.yaml for display
    exp_yaml_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    exp_yaml_content = ""
    if exp_yaml_path.exists():
        exp_yaml_content = exp_yaml_path.read_text(encoding="utf-8")

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
            "experimentYaml": exp_yaml_content,
            "factorLibraries": [Path(p).name for p in _find_factor_jsons()],
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

def _update_mining_metrics(task: Dict[str, Any]):
    """
    Update mining task metrics from the generated factor library.
    Calculates best factor stats and extracts top 10 factors.
    """
    jsons = _find_factor_jsons()
    # Prefer library with matching suffix if configured
    target_lib = None
    config = task.get("config", {})
    suffix = config.get("librarySuffix")
    
    if suffix:
        candidate = PROJECT_ROOT / "data" / "factorlib" / f"all_factors_library_{suffix}.json"
        # Fix: If suffix is specified, we ONLY look at this file.
        # If it doesn't exist yet, it means no factors have been mined yet for this task.
        if candidate.exists():
            target_lib = str(candidate)
        else:
            # Task specific file not found -> assume empty state
            return
            
    elif jsons:
        # No suffix provided, fallback to latest existing library (legacy behavior)
        target_lib = jsons[0]
        
    if not target_lib:
        return

    # Check modification time
    try:
        mtime = os.path.getmtime(target_lib)
        created_at_str = task.get("createdAt")
        if created_at_str:
            created_at_dt = datetime.fromisoformat(created_at_str)
            # Add a small buffer (e.g. 1 second) to avoid race conditions where file is created immediately
            if mtime < created_at_dt.timestamp():
                # File is older than the task -> ignore it
                return
    except Exception:
        pass

    try:
        lib = _load_factor_library(target_lib)
        factors = lib.get("factors", {})
        
        # 1. Update basic stats
        total = len(factors)
        task["metrics"]["totalFactors"] = total
        
        high = medium = low = 0
        factor_list = []
        
        for f_id, f_info in factors.items():
            # Check if this factor was created after task start
            # If we are using a shared library file (unlikely with new logic, but possible if user forces it),
            # we must ensure we don't display old factors.
            try:
                added_at_str = f_info.get("added_at", "")
                created_at_str = task.get("createdAt", "")
                if added_at_str and created_at_str:
                    # Parse timestamps
                    # added_at usually in isoformat
                    added_at_dt = datetime.fromisoformat(added_at_str)
                    created_at_dt = datetime.fromisoformat(created_at_str)
                    if added_at_dt < created_at_dt:
                        continue
            except Exception:
                pass # If date parsing fails, be permissive or conservative? Permissive for now.

            bt = f_info.get("backtest_results", {})
            q = _classify_quality(bt)
            if q == "high": high += 1
            elif q == "medium": medium += 1
            else: low += 1
            
            # Prepare for top 10 list
            # Normalize metrics
            ic = bt.get("IC", bt.get("1day.excess_return_without_cost.information_coefficient", 0))
            icir = bt.get("ICIR", bt.get("1day.excess_return_without_cost.information_coefficient_ir", 0))
            rank_ic = bt.get("Rank IC", bt.get("rank_ic", bt.get("1day.excess_return_without_cost.rank_ic", 0)))
            rank_icir = bt.get("Rank ICIR", bt.get("rank_ic_ir", bt.get("1day.excess_return_without_cost.rank_ic_ir", 0)))
            
            # Generate a mock equity curve for preview if real data is missing
            # In production, this should come from actual backtest result files (CSV/H5)
            # Here we generate a simple random walk with drift matching the annual return to show visual difference
            cumulative_curve = []
            annual_ret = bt.get("1day.excess_return_without_cost.annualized_return", 0)
            max_dd = bt.get("1day.excess_return_with_cost.max_drawdown", 
                                    bt.get("1day.excess_return_without_cost.max_drawdown", 0))
            
            # Calmar Ratio = Annual Return / Max Drawdown (absolute value)
            # Avoid division by zero
            cr = 0
            if max_dd < 0:
                cr = annual_ret / abs(max_dd)
            elif max_dd > 0:
                cr = annual_ret / max_dd
            
            # Simple simulation: 20 data points for preview sparkline
            import random
            current_val = 1.0
            # Daily drift approx
            drift = (1 + annual_ret) ** (1/252) - 1 if annual_ret else 0
            vol = 0.02 # Assumed daily vol
            
            # Use factor name hash to seed random for consistency
            random.seed(hash(f_info.get("factor_name", f_id)))
            
            for i in range(20):
                 # Generate last 20 points
                 ret = random.gauss(drift, vol)
                 current_val *= (1 + ret)
                 cumulative_curve.append({"value": current_val, "date": f"Day {i+1}"})
            
            factor_list.append({
                "factorName": f_info.get("factor_name", f_id),
                "factorExpression": f_info.get("factor_expression", ""),
                "rankIc": rank_ic,
                "rankIcir": rank_icir,
                "ic": ic,
                "icir": icir,
                "annualReturn": annual_ret,
                "sharpeRatio": bt.get("1day.excess_return_with_cost.information_ratio", 
                                    bt.get("1day.excess_return_without_cost.information_ratio", 0)),
                "maxDrawdown": max_dd,
                "calmarRatio": cr,
                "cumulativeCurve": cumulative_curve
            })

        task["metrics"]["highQualityFactors"] = high
        task["metrics"]["mediumQualityFactors"] = medium
        task["metrics"]["lowQualityFactors"] = low
        
        # 2. Find best factor
        if factor_list:
            # Sort by RankIC desc
            factor_list.sort(key=lambda x: x["rankIc"], reverse=True)
            best = factor_list[0]
            
            # Update task metrics with best factor's stats
            task["metrics"]["annualReturn"] = best["annualReturn"]
            task["metrics"]["rankIc"] = best["rankIc"]
            task["metrics"]["sharpeRatio"] = best["sharpeRatio"]
            task["metrics"]["maxDrawdown"] = best["maxDrawdown"]
            task["metrics"]["factorName"] = best["factorName"]
            
            # 3. Top 10 Factors
            task["metrics"]["top10Factors"] = factor_list[:10]
            
    except Exception:
        pass # Best effort

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")

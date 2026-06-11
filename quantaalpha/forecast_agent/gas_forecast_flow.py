"""Single-round orchestrated gas forecast flow (M1).

Pipeline:
1) parse intent from query
2) run xgboost baseline and collect feature importance
3) ask LLM to recommend one feature superset (importance-driven)
4) run multi-model auto selection with per-model feature projection
5) build decadal + monthly rollup summary
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantaalpha.forecast_agent.data import AS_OF_DATE
from quantaalpha.forecast_agent.feature_registry import FEATURE_REGISTRY, feature_pool_for_prompt
from quantaalpha.forecast_agent.feature_engineering import SARIMAX_EXOG_COLS
from quantaalpha.forecast_agent.lstm_agent import SELECTED_FEATURES
from quantaalpha.forecast_agent.runner import (
    DEFAULT_CANDIDATE_MODELS,
    ForecastRunConfig,
    run_forecast,
)
from quantaalpha.llm.client import APIBackend
from quantaalpha.log import logger

DEFAULT_AGENT_CANDIDATE_MODELS = (
    "catboost,elasticnet,lasso,lightgbm,lstm,random_forest,ridge,sarimax,xgboost"
)
FIXED_AS_OF_DATE = AS_OF_DATE


@dataclass(frozen=True)
class GasForecastFlowConfig:
    query: str
    province: str = "河北"
    as_of_month: str = FIXED_AS_OF_DATE
    context_len: int = 270
    candidate_models: str = DEFAULT_AGENT_CANDIDATE_MODELS
    output_dir: str | None = None
    max_feature_count: int = 10
    importance_top_k: int = 12
    required_features: list[str] | None = None
    qa_query: str | None = None


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("LLM 返回为空")
    start = raw.find("{")
    end = raw.rfind("}")
    snippet = raw[start : end + 1] if start >= 0 and end > start else raw
    payload = json.loads(snippet)
    if not isinstance(payload, dict):
        raise ValueError("LLM 返回必须是 JSON 对象")
    return payload


def _parse_target_month(query: str) -> str | None:
    text = str(query or "").strip()
    # 2026年3月 / 2026-03 / 2026/03
    m = re.search(r"(20\d{2})\s*[年/\-\.]\s*(1[0-2]|0?[1-9])\s*月?", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    return None


def _month_to_decadal_range(month_yyyy_mm: str) -> tuple[str, str]:
    y, m = month_yyyy_mm.split("-")
    return (f"{int(y):04d}-{int(m):02d}-01", f"{int(y):04d}-{int(m):02d}-21")


def _coerce_decadal_date(raw: Any) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    day = int(ts.day)
    if day not in (1, 11, 21):
        return None
    return f"{int(ts.year):04d}-{int(ts.month):02d}-{day:02d}"


def _normalize_province_name(raw: Any, *, default_province: str) -> str:
    text = str(raw or "").strip()
    if not text:
        text = str(default_province or "").strip()
    if not text:
        return "河北"
    for suffix in ("维吾尔自治区", "壮族自治区", "回族自治区", "特别行政区", "自治区", "省", "市"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    return text or str(default_province or "河北").strip() or "河北"


def _parse_intent_with_llm(query: str, *, default_province: str) -> dict[str, Any]:
    system_prompt = (
        "你是天然气预测参数解析助手。"
        "请把用户自然语言需求解析为结构化参数。"
        "只输出一个严格 JSON 对象，不要输出任何额外文字。"
    )
    user_prompt = f"""请解析下面的需求文本，提取预测参数：
{query}

输出 JSON 字段（全部字段必须出现，未知可填 null）：
{{
  "province": "省份中文名（尽量不带'省/市/自治区'后缀）或null",
  "target_month": "YYYY-MM 或 null",
  "test_start": "YYYY-MM-DD 或 null",
  "test_end": "YYYY-MM-DD 或 null"
}}

要求：
1) 若文本只提到“某年某月”，优先填写 target_month；
2) test_start/test_end 若填写，必须是旬开始日（1/11/21）；
3) 若原文未提及具体日期，可填 null，不要臆造；
4) 省份字段尽量输出简称，如“河北”“北京”“内蒙古”“广西”；
5) 不要编造超出原文的信息。
"""
    raw = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        reasoning_flag=False,
    )
    payload = _extract_json_object(raw)
    return {
        "province": _normalize_province_name(payload.get("province"), default_province=default_province),
        "target_month": str(payload.get("target_month") or "").strip() or None,
        "test_start": _coerce_decadal_date(payload.get("test_start")),
        "test_end": _coerce_decadal_date(payload.get("test_end")),
    }


def parse_intent(query: str, *, default_province: str, as_of_month: str | None = None) -> dict[str, str]:
    """Parse forecasting intent with explicit date rules.

    Rules:
    1) target_month:
       - Prefer LLM parsed value.
       - Else try regex extraction from query.
       - If still missing -> raise (no silent default month).
    2) test_start/test_end:
       - If both are provided and valid decadal dates, use them.
       - If both are missing, derive from target_month: YYYY-MM-01 ~ YYYY-MM-21.
       - If only one side is provided -> raise.
    3) as_of_month:
       - Fixed to the configured as_of date (no LLM parse, no rule fallback).
    """
    llm_obj: dict[str, Any] | None = None
    llm_err: Exception | None = None
    try:
        llm_obj = _parse_intent_with_llm(query, default_province=default_province)
    except Exception as exc:  # noqa: BLE001
        llm_err = exc
        logger.warning(f"parse_intent LLM 解析失败，尝试规则解析: {exc}")

    target_month = (llm_obj or {}).get("target_month") or _parse_target_month(query)
    if not target_month:
        if llm_err is not None:
            raise ValueError(
                "无法解析目标月份：LLM 解析失败且 query 未提供明确月份，请在 query 中写明 YYYY年M月"
            ) from llm_err
        raise ValueError("无法从 query 解析目标月份，请明确说明如“预测 2026年3月 河北天然气销量”")
    test_start = (llm_obj or {}).get("test_start")
    test_end = (llm_obj or {}).get("test_end")
    if test_start and test_end:
        pass
    elif (test_start and not test_end) or (test_end and not test_start):
        raise ValueError("test_start/test_end 需同时提供，或都不提供由 target_month 自动推导")
    else:
        test_start, test_end = _month_to_decadal_range(target_month)
    inferred_as_of = _coerce_decadal_date(as_of_month)
    if not inferred_as_of:
        raise ValueError(
            f"as_of_month 必须是旬开始日（1/11/21），当前值: {as_of_month!r}"
        )
    province = _normalize_province_name((llm_obj or {}).get("province"), default_province=default_province)
    return {
        "query": query,
        "province": province,
        "target_month": target_month,
        "test_start": test_start,
        "test_end": test_end,
        "as_of_month": inferred_as_of,
    }


def _sorted_importance_items(importance: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[tuple[str, float]] = []
    for k, v in (importance or {}).items():
        try:
            rows.append((str(k), float(v)))
        except Exception:
            continue
    rows.sort(key=lambda x: x[1], reverse=True)
    return [{"name": name, "score": score} for name, score in rows]


def _normalize_candidate_models(raw: Any) -> str:
    """Normalize CLI/Fire candidate_models into comma-separated model names.

    Fire may parse "xgboost,lasso" as a tuple-like value, so accept both strings
    and sequences here.
    """
    if raw in (None, "", "null"):
        return DEFAULT_AGENT_CANDIDATE_MODELS
    if isinstance(raw, (list, tuple, set)):
        items = [str(x).strip().strip("'\"") for x in raw if str(x).strip()]
    else:
        text = str(raw).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        items = [x.strip().strip("'\"") for x in text.split(",") if x.strip()]
    deduped: list[str] = []
    for item in items:
        model = item.lower()
        if model and model not in deduped:
            deduped.append(model)
    return ",".join(deduped) if deduped else DEFAULT_AGENT_CANDIDATE_MODELS


def _project_feature_superset_for_model(model: str, feature_superset: list[str]) -> list[str]:
    """Project one shared feature superset into per-model allowed feature_set.

    This is the core fix for mixed-model runs: LSTM/SARIMAX have strict input
    capabilities and cannot directly consume tabular full superset.
    """
    model_key = str(model or "").strip().lower()
    deduped = []
    for col in feature_superset:
        name = str(col).strip()
        if name and name not in deduped:
            deduped.append(name)

    if model_key == "lstm":
        allowed = set(SELECTED_FEATURES)
        return [c for c in deduped if c in allowed]
    if model_key == "sarimax":
        allowed = set(SARIMAX_EXOG_COLS)
        return [c for c in deduped if c in allowed]
    if model_key == "timesfm":
        return []
    return deduped


def _project_feature_superset_by_model(models: list[str], feature_superset: list[str]) -> dict[str, list[str]]:
    return {m: _project_feature_superset_for_model(m, feature_superset) for m in models}


def diagnose_importance(intent: dict[str, str], *, output_dir: Path, context_len: int) -> dict[str, Any]:
    diag_dir = output_dir / "_diagnose_xgboost"
    cfg = ForecastRunConfig(
        model="xgboost",
        province=intent["province"],
        as_of_month=intent["as_of_month"],
        test_start=intent["test_start"],
        test_end=intent["test_end"],
        output_dir=str(diag_dir),
        context_len=context_len,
    )
    run_forecast(cfg)

    summary_path = diag_dir / "xgboost" / "xgboost_best_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到 xgboost baseline summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    importance = summary.get("feature_importance") or {}
    ranked = _sorted_importance_items(importance)
    return {
        "diagnose_dir": str(diag_dir),
        "summary_json": str(summary_path),
        "importance_items": ranked,
        "feature_cols_used": list(summary.get("feature_cols_used") or []),
        "test_metrics": summary.get("test_metrics") or {},
    }


def recommend_feature_superset(
    *,
    query: str,
    importance_items: list[dict[str, Any]],
    max_feature_count: int,
    top_k: int,
    required_features: list[str] | None = None,
) -> dict[str, Any]:
    top_items = importance_items[: max(1, int(top_k))]
    feature_pool_json = feature_pool_for_prompt()
    required = [f for f in (required_features or ["HDD", "Lag_36"]) if f in FEATURE_REGISTRY]

    system_prompt = (
        "你是燃气销量预测特征选择助手。"
        "你必须基于特征重要性证据推荐特征。"
        "禁止把测试集MAPE排行作为主要推荐依据。"
        "输出必须是严格 JSON 对象。"
    )
    user_prompt = f"""请基于以下信息推荐一组特征（单个方案）。\n
【用户目标】
{query}

【xgboost 特征重要性 Top 列表】
{json.dumps(top_items, ensure_ascii=False)}

【可选特征池】（只能从中选择）
{feature_pool_json}

硬性约束：
1. 返回一个 JSON 对象，不要 markdown，不要额外解释；
2. feature_set 只能使用特征池中的名称；
3. feature_set 数量 <= {int(max_feature_count)};
4. 推荐依据应优先使用重要性证据，不以测试集 MAPE 排行作为主依据；
5. 必须包含特征：{required}。

输出格式：
{{
  "feature_set": ["..."],
  "reason": "..."
}}
"""
    raw = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        reasoning_flag=False,
    )
    payload = _extract_json_object(raw)
    picked = [str(x) for x in (payload.get("feature_set") or []) if str(x) in FEATURE_REGISTRY]
    reason = str(payload.get("reason") or "").strip()

    # ensure required anchors
    ordered: list[str] = []
    for col in [*required, *picked]:
        if col in FEATURE_REGISTRY and col not in ordered:
            ordered.append(col)
    if not ordered:
        raise ValueError("LLM 返回的 feature_set 为空或均不在特征池内，请修正 prompt 或模型输出")
    ordered = ordered[: max(1, int(max_feature_count))]
    return {
        "feature_superset": ordered,
        "reason": reason,
        "source_top_items": top_items,
    }


def _load_model_feature_cols_used(out_root: Path, model: str) -> list[str]:
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


def _monthly_rollup_from_model_dir(model_dir: Path, model: str) -> list[dict[str, Any]]:
    table_xlsx = model_dir / f"{model}_test.xlsx"
    table_csv = model_dir / f"{model}_test.csv"
    if table_xlsx.exists():
        df = pd.read_excel(table_xlsx)
    elif table_csv.exists():
        df = pd.read_csv(table_csv)
    else:
        raise FileNotFoundError(f"未找到月度汇总数据源: {table_xlsx} / {table_csv}")

    if "date" in df.columns:
        ds = pd.to_datetime(df["date"], errors="coerce")
    elif "ds" in df.columns:
        ds = pd.to_datetime(df["ds"], errors="coerce")
    else:
        raise ValueError(f"{model}_test 表缺少 date/ds 列，无法做月度汇总")

    yhat_col = "predicted_gas_sales" if "predicted_gas_sales" in df.columns else "yhat"
    if yhat_col not in df.columns:
        raise ValueError(f"{model}_test 表缺少 predicted_gas_sales/yhat 列，无法做月度汇总")
    y_col = "actual_gas_sales" if "actual_gas_sales" in df.columns else None

    work = pd.DataFrame(
        {
            "month": ds.dt.strftime("%Y-%m"),
            "yhat": pd.to_numeric(df[yhat_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce") if y_col else pd.Series([pd.NA] * len(df)),
        }
    ).dropna(subset=["month", "yhat"])
    if work.empty:
        raise ValueError(f"{model}_test 表中无可用的月度聚合样本")

    grouped = work.groupby("month", as_index=False).agg({"yhat": "sum", "y": "sum"})
    rows: list[dict[str, Any]] = []
    for _, r in grouped.iterrows():
        rec: dict[str, Any] = {
            "month": str(r["month"]),
            "predicted_gas_sales": float(r["yhat"]),
        }
        yv = r["y"]
        if pd.notna(yv):
            rec["actual_gas_sales"] = float(yv)
            if abs(float(yv)) > 1e-12:
                rec["mape_pct"] = abs((float(r["yhat"]) - float(yv)) / float(yv)) * 100.0
            else:
                rec["mape_pct"] = None
        rows.append(rec)
    return rows


def _load_qa_rows(out_root: Path, model: str) -> list[dict[str, Any]]:
    model_dir = out_root / model
    test_xlsx = model_dir / f"{model}_test.xlsx"
    test_csv = model_dir / f"{model}_test.csv"
    forecast_csv = model_dir / f"{model}_forecast.csv"
    if test_xlsx.exists():
        df = pd.read_excel(test_xlsx)
    elif test_csv.exists():
        df = pd.read_csv(test_csv)
    elif forecast_csv.exists():
        df = pd.read_csv(forecast_csv)
    else:
        return []

    ds_col = "date" if "date" in df.columns else ("ds" if "ds" in df.columns else None)
    yhat_col = "predicted_gas_sales" if "predicted_gas_sales" in df.columns else ("yhat" if "yhat" in df.columns else None)
    if not ds_col or not yhat_col:
        return []
    y_col = "actual_gas_sales" if "actual_gas_sales" in df.columns else ("y" if "y" in df.columns else None)
    period_col = "month" if "month" in df.columns else None

    out_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        yhat = pd.to_numeric(r.get(yhat_col), errors="coerce")
        if pd.isna(yhat):
            continue
        y = pd.to_numeric(r.get(y_col), errors="coerce") if y_col else pd.NA
        y_val = None if pd.isna(y) else float(y)
        yhat_val = float(yhat)
        err = None
        if y_val is not None and abs(y_val) > 1e-12:
            err = (yhat_val - y_val) / y_val * 100.0
        out_rows.append(
            {
                "ds": str(r.get(ds_col)),
                "period": str(r.get(period_col)) if period_col else "",
                "yhat": yhat_val,
                "y": y_val,
                "error_pct": err,
            }
        )
    return out_rows


def ask_flow_qa(
    *,
    output_dir: str | Path,
    model: str,
    query: str,
    selected_features: list[str] | None = None,
) -> dict[str, Any]:
    from quantaalpha.app.utils.forecast_qa import generate_forecast_qa_answer

    out_root = Path(output_dir)
    model_name = str(model or "").strip().lower()
    if not model_name:
        raise ValueError("model 不能为空")
    rows = _load_qa_rows(out_root, model_name)
    if not rows:
        raise FileNotFoundError(f"未找到可问答的数据表: {out_root / model_name}")

    return generate_forecast_qa_answer(
        query=query,
        rows=rows,
        feature_cols_used=_load_model_feature_cols_used(out_root, model_name),
        selected_features=list(selected_features or []),
    )


def run_compare_and_rollup(
    *,
    intent: dict[str, str],
    feature_superset: list[str],
    output_dir: Path,
    context_len: int,
    candidate_models: str,
) -> dict[str, Any]:
    candidate_models = _normalize_candidate_models(candidate_models)
    models = [m.strip().lower() for m in candidate_models.split(",") if m.strip()]
    if not models:
        models = [m.strip() for m in DEFAULT_CANDIDATE_MODELS.split(",") if m.strip()]
    per_model_features = _project_feature_superset_by_model(models, feature_superset)

    run_cfg = ForecastRunConfig(
        model="auto",
        province=intent["province"],
        as_of_month=intent["as_of_month"],
        test_start=intent["test_start"],
        test_end=intent["test_end"],
        output_dir=str(output_dir),
        context_len=context_len,
        candidate_models=",".join(models),
        model_features=per_model_features,
    )
    auto_result = run_forecast(run_cfg)

    best_model = str(auto_result.get("best_model") or "").strip().lower()
    candidates = list(auto_result.get("candidates") or [])
    leaderboard: list[dict[str, Any]] = []
    for item in candidates:
        backend = str(item.get("backend") or "").strip().lower()
        tm = item.get("test_metrics") or {}
        leaderboard.append(
            {
                "model": backend,
                "MAPE": tm.get("MAPE"),
                "RMSE": tm.get("RMSE"),
                "MAE": tm.get("MAE"),
                "R2": tm.get("R2"),
                "feature_cols_used": _load_model_feature_cols_used(output_dir, backend),
            }
        )
    leaderboard.sort(key=lambda x: (float(x["MAPE"]) if x.get("MAPE") is not None else float("inf")))
    for idx, row in enumerate(leaderboard, start=1):
        row["rank"] = idx

    monthly_rollup = _monthly_rollup_from_model_dir(output_dir / best_model, best_model) if best_model else []
    return {
        "best_model": best_model,
        "leaderboard": leaderboard,
        "monthly_rollup": monthly_rollup,
    }


def run_gas_forecast_flow(config: GasForecastFlowConfig) -> dict[str, Any]:
    if not str(config.query).strip():
        raise ValueError("query 不能为空")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(config.output_dir or f"forecast_agent_output/{config.province}/agent_flow_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    intent = parse_intent(config.query, default_province=config.province, as_of_month=config.as_of_month)
    diagnose = diagnose_importance(intent, output_dir=out_root, context_len=config.context_len)
    recommend = recommend_feature_superset(
        query=config.query,
        importance_items=diagnose["importance_items"],
        max_feature_count=config.max_feature_count,
        top_k=config.importance_top_k,
        required_features=config.required_features,
    )
    compare = run_compare_and_rollup(
        intent=intent,
        feature_superset=recommend["feature_superset"],
        output_dir=out_root,
        context_len=config.context_len,
        candidate_models=config.candidate_models,
    )

    payload = {
        "query": config.query,
        "intent": intent,
        "diagnose": {
            "summary_json": diagnose["summary_json"],
            "feature_importance_top": diagnose["importance_items"][: config.importance_top_k],
            "test_metrics": diagnose["test_metrics"],
        },
        "recommend": {
            "feature_superset": recommend["feature_superset"],
            "reason": recommend["reason"],
        },
        "compare": compare,
        "output_dir": str(out_root),
    }
    if str(config.qa_query or "").strip() and compare.get("best_model"):
        payload["qa"] = ask_flow_qa(
            output_dir=out_root,
            model=str(compare["best_model"]),
            query=str(config.qa_query),
            selected_features=list(recommend["feature_superset"]),
        )
    result_json = out_root / "agent_flow_result.json"
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def gas_forecast_flow_from_fire(**kwargs: Any) -> dict[str, Any]:
    if kwargs.get("as_of_month") not in (None, "", FIXED_AS_OF_DATE):
        raise ValueError(
            f"forecast_flow 当前约束 as_of_month 固定为 {FIXED_AS_OF_DATE}，"
            "请不要在命令行覆盖。"
        )
    cfg = GasForecastFlowConfig(
        query=str(kwargs.get("query") or ""),
        province=str(kwargs.get("province") or "河北"),
        as_of_month=FIXED_AS_OF_DATE,
        context_len=int(kwargs.get("context_len") or 270),
        candidate_models=_normalize_candidate_models(kwargs.get("candidate_models")),
        output_dir=(str(kwargs.get("output_dir")) if kwargs.get("output_dir") else None),
        max_feature_count=int(kwargs.get("max_feature_count") or 10),
        importance_top_k=int(kwargs.get("importance_top_k") or 12),
        qa_query=(str(kwargs.get("qa_query")) if kwargs.get("qa_query") else None),
        required_features=(
            [str(x) for x in kwargs.get("required_features", [])]
            if isinstance(kwargs.get("required_features"), list)
            else None
        ),
    )
    return run_gas_forecast_flow(cfg)


__all__ = [
    "DEFAULT_AGENT_CANDIDATE_MODELS",
    "FIXED_AS_OF_DATE",
    "GasForecastFlowConfig",
    "parse_intent",
    "diagnose_importance",
    "recommend_feature_superset",
    "_normalize_candidate_models",
    "ask_flow_qa",
    "run_compare_and_rollup",
    "run_gas_forecast_flow",
    "gas_forecast_flow_from_fire",
]


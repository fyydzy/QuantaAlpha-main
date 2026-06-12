"""燃气预测编排层（第三版核心）。

Web 与 CLI 共用本模块的业务函数；Web 在 app.py 的 _run_forecast_agent 里按阶段调用，
并在两个人机确认点之间插入 _wait_for_continue。

Pipeline:
1) parse_intent — LLM 解析 query → 结构化 intent
2) diagnose_importance — 全特征 xgboost，产出 feature_importance（与最终特征无关）
3) recommend_feature_superset — LLM 推荐特征超集
4) run_compare_and_rollup — 逐模型投影 + auto 比选 + 月度汇总
5) ask_flow_qa（可选）— 读预测表问答

人机确认覆写：parse_continue_message → normalize_continue_overrides → apply_intent_overrides
字段提取仅 LLM；特征池 FEATURE_REGISTRY 仅作校验过滤。
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
    """省份简称归一化（去省/市/自治区后缀）。不负责判断用户是否提及省份，那由 LLM prompt 决定。"""
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
    """首轮意图：调 LLM，prompt 内含单月/跨月示例。不做正则从 query 抠字段。"""
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
1) 若文本只提到“某年某月”，优先填写 target_month，test_start/test_end 填 null；
2) 若文本提到跨月区间（如“3月到4月”“2026年3月至4月”），必须同时填写 test_start 和 test_end；
   target_month 填区间结束月（如 2026-04）；
3) test_start/test_end 若填写，必须是旬开始日（1/11/21）；
4) 若原文未提及具体日期，必须填 null，不要臆造；
5) province 只有在用户明确提及省份时才可填写；未提及必须填 null；
6) province 只能是行政区名称，不得输出无关词汇；
7) 省份字段一定要输出简称，如“河北”“北京”“内蒙古”“广西”；
8) 不要编造超出原文的信息；
9) 输出前自检：四个字段都必须存在；未提及字段必须为 null。

示例1（单月）：
输入：预测2026年3月河北天然气销量
输出：
{{
  "province": "河北",
  "target_month": "2026-03",
  "test_start": null,
  "test_end": null
}}

示例2（跨月）：
输入：预测2026年3月到4月河北天然气销量
输出：
{{
  "province": "河北",
  "target_month": "2026-04",
  "test_start": "2026-03-01",
  "test_end": "2026-04-21"
}}
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
    """从用户 query 解析预测 intent（Web 阶段 0 / CLI 第一步）。

    规则摘要：
    - target_month 必须来自 LLM，缺失则报错（无正则兜底）
    - test_start/end：LLM 同时给出则用；都缺则从 target_month 推当月 01~21；只给一侧报错
    - as_of_month 固定 FIXED_AS_OF_DATE，不由 query 解析
    """
    try:
        llm_obj = _parse_intent_with_llm(query, default_province=default_province)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"parse_intent LLM 解析失败（无规则兜底）: {exc}")
        raise ValueError("无法解析目标月份：LLM 解析失败，请重试并明确说明月份。") from exc

    target_month = (llm_obj or {}).get("target_month")
    if not target_month:
        raise ValueError("无法解析目标月份：请在需求中明确说明如“预测 2026年3月 河北天然气销量”")
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


def apply_intent_overrides(intent: dict[str, str], overrides: dict[str, Any]) -> dict[str, str]:
    """合并 confirm_intent 覆写：null/空字段不改动；只改 target_month 时自动同步 test 区间。"""
    if not overrides:
        return intent

    updated = dict(intent)
    for key, value in overrides.items():
        if value is not None and str(value).strip():
            updated[key] = str(value).strip()

    default_province = str(intent.get("province") or "河北")
    if updated.get("province"):
        updated["province"] = _normalize_province_name(updated["province"], default_province=default_province)

    test_start_explicit = bool(str(overrides.get("test_start") or "").strip())
    test_end_explicit = bool(str(overrides.get("test_end") or "").strip())
    target_month_explicit = bool(str(overrides.get("target_month") or "").strip())

    if test_start_explicit != test_end_explicit:
        raise ValueError("test_start/test_end 需同时提供，或都不提供由 target_month 自动推导")

    if target_month_explicit and not (test_start_explicit and test_end_explicit):
        ts, te = _month_to_decadal_range(updated["target_month"])
        updated["test_start"] = ts
        updated["test_end"] = te
    elif test_start_explicit and test_end_explicit:
        ts = _coerce_decadal_date(updated.get("test_start"))
        te = _coerce_decadal_date(updated.get("test_end"))
        if not ts or not te:
            raise ValueError("test_start/test_end 必须是旬开始日（1/11/21）")
        updated["test_start"] = ts
        updated["test_end"] = te

    return updated


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
    """阶段 1：全特征 xgboost 基线，读取 feature_importance 供推荐用（不用用户确认的特征集）。"""
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
    intent: dict[str, str] | None = None,
) -> dict[str, Any]:
    """阶段 2：LLM 结合重要性 Top-K + 特征池推荐一组 feature_superset（须传入已确认 intent）。"""
    top_items = importance_items[: max(1, int(top_k))]
    feature_pool_json = feature_pool_for_prompt()
    required = [f for f in (required_features or ["HDD", "Lag_36"]) if f in FEATURE_REGISTRY]

    intent_block = ""
    if intent:
        intent_block = f"""
【当前预测参数（已确认）】
省份: {intent.get("province")}
目标月份: {intent.get("target_month")}
测试区间: {intent.get("test_start")} ~ {intent.get("test_end")}
"""

    system_prompt = (
        "你是燃气销量预测特征选择助手。"
        "你必须基于特征重要性证据推荐特征。"
        "禁止把测试集MAPE排行作为主要推荐依据。"
        "输出必须是严格 JSON 对象。"
    )
    user_prompt = f"""请基于以下信息推荐一组特征（单个方案）。\n
【用户目标】
{query}
{intent_block}
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


def normalize_continue_overrides(checkpoint: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 返回的 overrides 整理为扁平字段；confirm_features 时过滤 FEATURE_REGISTRY（校验非提取）。"""
    raw = dict(overrides or {})
    if isinstance(raw.get("intent"), dict):
        raw = {**raw.get("intent"), **raw}

    if checkpoint == "confirm_intent":
        out: dict[str, Any] = {}
        for key, aliases in {
            "province": ("province",),
            "target_month": ("target_month", "targetMonth"),
            "test_start": ("test_start", "testStart"),
            "test_end": ("test_end", "testEnd"),
        }.items():
            for alias in aliases:
                val = raw.get(alias)
                if val is not None and str(val).strip():
                    out[key] = str(val).strip()
                    break
        return out

    if checkpoint == "confirm_features":
        fs = raw.get("featureSuperset")
        if fs is None:
            fs = raw.get("feature_superset")
        if isinstance(fs, str):
            fs = [x.strip() for x in re.split(r"[,，、;\s]+", fs) if x.strip()]
        if isinstance(fs, list):
            cleaned = [str(x).strip() for x in fs if str(x).strip() in FEATURE_REGISTRY]
            return {"featureSuperset": cleaned} if cleaned else {}
        return {}

    return raw


def _continue_intent_prompt(
    pending_payload: dict[str, Any],
    user_message: str,
) -> tuple[str, str]:
    intent = pending_payload.get("intent") if isinstance(pending_payload.get("intent"), dict) else pending_payload
    system_prompt = (
        "你是天然气预测参数解析助手。"
        "当前处于参数确认环节：用户已看到系统解析结果，正在用自然语言确认或修改。"
        "请判断用户是否同意继续，并仅提取用户明确要求修改的字段。"
        "只输出一个严格 JSON 对象，不要输出任何额外文字。"
    )
    user_prompt = f"""【当前已解析参数】
{json.dumps(intent or {}, ensure_ascii=False, indent=2)}

【用户回复】
{user_message}

请输出 JSON（全部顶层字段必须出现）：
{{
  "approved": true,
  "overrides": {{
    "province": "省份中文名（尽量不带'省/市/自治区'后缀）或 null",
    "target_month": "YYYY-MM 或 null",
    "test_start": "YYYY-MM-DD 或 null",
    "test_end": "YYYY-MM-DD 或 null"
  }}
}}

要求：
1) 用户明确拒绝/取消/停止/不同意 -> approved=false；否则 approved=true；
2) overrides 只填写用户明确要求修改的字段，未提及的字段填 null；
3) 用户仅说“继续/好的/确认/没问题”等表示同意时，overrides 各字段均为 null；
4) 若用户只改“某年某月”，只填 target_month，test_start/test_end 填 null；
5) 若用户要求跨月区间（如“改成3月到4月”），必须同时填 test_start 和 test_end，
   target_month 填区间结束月；未改动的字段填 null；
6) test_start/test_end 若填写，必须是旬开始日（1/11/21）；未提及则填 null，不要臆造；
7) 省份尽量输出简称，如“河北”“北京”“内蒙古”“广西”；
8) province 只有在用户明确提出“改省份/换地区”时才可填写，否则必须为 null；
9) province 只能是行政区名称，不得输出无关词汇；
10) 不要编造超出用户回复的信息；
11) 输出前自检：approved 必须是布尔；overrides 四个字段必须都出现，未提及字段必须为 null。

示例1（改单月）：
【当前已解析参数】
{{"province": "河北", "target_month": "2026-03", "test_start": "2026-03-01", "test_end": "2026-03-21"}}
【用户回复】
改成2025年12月
输出：
{{
  "approved": true,
  "overrides": {{
    "province": null,
    "target_month": "2025-12",
    "test_start": null,
    "test_end": null
  }}
}}

示例2（改跨月）：
【当前已解析参数】
{{"province": "河北", "target_month": "2026-03", "test_start": "2026-03-01", "test_end": "2026-03-21"}}
【用户回复】
改成预测2026年3月到4月
输出：
{{
  "approved": true,
  "overrides": {{
    "province": null,
    "target_month": "2026-04",
    "test_start": "2026-03-01",
    "test_end": "2026-04-21"
  }}
}}
"""
    return system_prompt, user_prompt


def _continue_features_prompt(
    pending_payload: dict[str, Any],
    user_message: str,
) -> tuple[str, str]:
    feature_pool_json = feature_pool_for_prompt()
    current_fs = pending_payload.get("featureSuperset") or pending_payload.get("feature_superset") or []
    reason = str(pending_payload.get("reason") or "").strip()
    system_prompt = (
        "你是燃气销量预测特征选择助手。"
        "当前处于特征确认环节：用户已看到系统推荐的特征组合，正在用自然语言确认或调整。"
        "请判断用户是否同意继续，并仅提取用户希望使用的特征组合。"
        "featureSuperset 只能使用特征池中的名称。"
        "输出必须是严格 JSON 对象，不要 markdown，不要额外解释。"
    )
    user_prompt = f"""【当前推荐特征组合】
{json.dumps(list(current_fs), ensure_ascii=False)}

【推荐理由】
{reason or "（无）"}

【可选特征池】（只能从中选择）
{feature_pool_json}

【用户回复】
{user_message}

请输出 JSON（全部顶层字段必须出现）：
{{
  "approved": true,
  "overrides": {{
    "featureSuperset": ["..."] 或 null
  }}
}}

硬性约束：
1) 用户明确拒绝/取消/停止/不同意 -> approved=false；否则 approved=true；
2) 用户仅说“继续/好的/确认/没问题”等表示同意时，overrides.featureSuperset 为 null；
3) 用户要求增删改特征时，输出完整的新 featureSuperset 数组（不是增量补丁）；
4) featureSuperset 只能使用特征池中的 name，禁止编造池外特征；
5) 若用户只说“去掉某特征”或“加上某特征”，在现有组合基础上调整后输出完整列表；
6) 用户未明确要求调整特征时，必须输出 null，不要猜测；
7) 不要编造用户未表达的特征偏好；
8) 输出前自检：approved 必须是布尔；overrides 中必须有且仅有 featureSuperset 字段。
"""
    return system_prompt, user_prompt


def _parse_continue_message_with_llm(
    checkpoint: str,
    pending_payload: dict[str, Any],
    user_message: str,
) -> dict[str, Any]:
    """人机确认：LLM 解析 approved + overrides；丢弃 null/空字段，不臆造未提及项。"""
    if checkpoint == "confirm_intent":
        system_prompt, user_prompt = _continue_intent_prompt(pending_payload, user_message)
    elif checkpoint == "confirm_features":
        system_prompt, user_prompt = _continue_features_prompt(pending_payload, user_message)
    else:
        system_prompt = (
            "你是预测任务的确认解析助手。"
            "请根据用户回复判断是否同意继续，并提取用户希望覆盖的参数。"
            "只输出严格 JSON 对象。"
        )
        user_prompt = f"""当前等待确认点：{checkpoint}
待确认内容（JSON）：
{json.dumps(pending_payload or {}, ensure_ascii=False)}
用户回复：
{user_message}

请输出：{{"approved": true/false, "overrides": {{}}}}
"""

    raw = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        reasoning_flag=False,
    )
    parsed = _extract_json_object(raw)
    approved = bool(parsed.get("approved", True))
    raw_overrides = parsed.get("overrides") if isinstance(parsed.get("overrides"), dict) else {}

    if checkpoint == "confirm_intent":
        cleaned: dict[str, Any] = {}
        for key in ("province", "target_month", "test_start", "test_end"):
            val = raw_overrides.get(key)
            if val is None or not str(val).strip():
                continue
            cleaned[key] = str(val).strip()
        raw_overrides = cleaned
    elif checkpoint == "confirm_features":
        fs = raw_overrides.get("featureSuperset")
        if fs is None:
            fs = raw_overrides.get("feature_superset")
        if fs is not None:
            if isinstance(fs, str):
                fs = [x.strip() for x in re.split(r"[,，、;\s]+", fs) if x.strip()]
            if isinstance(fs, list):
                picked = [str(x).strip() for x in fs if str(x).strip() in FEATURE_REGISTRY]
                raw_overrides = {"featureSuperset": picked} if picked else {}
            else:
                raw_overrides = {}

    overrides = normalize_continue_overrides(checkpoint, raw_overrides)
    return {"approved": approved, "overrides": overrides}


def parse_continue_message(
    checkpoint: str,
    pending_payload: dict[str, Any],
    user_message: str,
    *,
    req_overrides: dict[str, Any] | None = None,
    req_approved: bool = True,
) -> dict[str, Any]:
    """Web continue 入口：仅 LLM 解析用户回复；失败抛 ValueError，无规则兜底。"""
    approved = bool(req_approved)
    overrides = normalize_continue_overrides(checkpoint, req_overrides or {})

    text = str(user_message or "").strip()
    if not text:
        return {"approved": approved, "overrides": overrides}

    try:
        llm_parsed = _parse_continue_message_with_llm(checkpoint, pending_payload, text)
        approved = bool(llm_parsed.get("approved", approved))
        llm_overrides = llm_parsed.get("overrides")
        if isinstance(llm_overrides, dict) and llm_overrides:
            overrides = {**overrides, **llm_overrides}
    except Exception as exc:
        logger.warning(f"parse_continue_message LLM 解析失败（无规则兜底）: {exc}")
        raise ValueError("继续确认解析失败，请重试。") from exc

    return {"approved": approved, "overrides": overrides}


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
    """阶段 3：按模型投影特征 → runner auto 比选 → 月度 rollup。"""
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
    """CLI 一条龙：无两个人机确认点，顺序调用各阶段函数。"""
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
    "apply_intent_overrides",
    "diagnose_importance",
    "recommend_feature_superset",
    "normalize_continue_overrides",
    "parse_continue_message",
    "_normalize_candidate_models",
    "ask_flow_qa",
    "run_compare_and_rollup",
    "run_gas_forecast_flow",
    "gas_forecast_flow_from_fire",
]


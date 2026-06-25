from __future__ import annotations

import json
import os
from typing import Any

from quantaalpha.forecast_agent.llm.client_flow import create_chat_completion_with_retry
from quantaalpha.llm.config import LLM_SETTINGS


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _ensure_qa_enabled() -> None:
    if not _env_flag("FORECAST_QA_ENABLED", default=True):
        raise RuntimeError("FORECAST_QA_ENABLED=false，问答功能已禁用")

    api_key = (
        LLM_SETTINGS.chat_openai_api_key
        or LLM_SETTINGS.openai_api_key
        or os.getenv("CHAT_OPENAI_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    if not str(api_key).strip():
        raise RuntimeError("未配置 OPENAI_API_KEY，无法调用 LLM 问答")


def generate_forecast_qa_answer(
    *,
    query: str,
    rows: list[dict[str, Any]],
    feature_cols_used: list[str] | None = None,
    selected_features: list[str] | None = None,
) -> dict[str, Any]:
    _ensure_qa_enabled()
    api_key = (
        LLM_SETTINGS.chat_openai_api_key
        or LLM_SETTINGS.openai_api_key
        or os.getenv("CHAT_OPENAI_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )

    clean_query = (query or "").strip()
    if not clean_query:
        raise ValueError("query 不能为空")
    if len(clean_query) > 1000:
        raise ValueError("query 过长，请控制在 1000 字以内")

    max_rows = _env_int("FORECAST_QA_MAX_ROWS", 120)
    limited_rows = list(rows)[: max_rows if max_rows > 0 else 120]

    feature_cols_used = list(feature_cols_used or [])
    selected_features = list(selected_features or [])
    data_mode = "test_only" if any(r.get("y") is not None for r in limited_rows) else "pred_only"

    system_prompt = (
        "你是燃气预测结果分析助手。"
        "只能依据给定表格和特征信息回答，不得编造外部事实。"
        "请优先给出结论，再说明依据（具体旬、误差幅度、趋势变化）。"
        "若没有真实值，请明确说明“仅基于预测值，无法判断真实偏差”。"
    )

    user_payload = {
        "query": clean_query,
        "data_mode": data_mode,
        "feature_priority_rule": "若 selected_features 与 feature_cols_used 不一致，以 feature_cols_used 为准",
        "selected_features": selected_features,
        "feature_cols_used": feature_cols_used,
        "columns": {
            "ds": "旬日期",
            "period": "旬标签（可空）",
            "yhat": "预测值",
            "y": "真实值（可空）",
            "error_pct": "(预测-真实)/真实*100，百分比（可空）",
        },
        "rows": limited_rows,
        "output_requirements": [
            "用中文回答",
            "先给3-6条要点结论",
            "最后给1条可执行建议",
        ],
    }

    answer = create_chat_completion_with_retry(
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        system_prompt=system_prompt,
        chat_api_key=api_key or None,
        scene="forecast_qa",
        json_mode=False,
        reasoning_flag=False,
        temperature=0.2,
        max_tokens=1200,
    )

    return {
        "answer": (answer or "").strip(),
        "model_used": str(LLM_SETTINGS.chat_model),
        "rows_used": len(limited_rows),
        "data_mode": data_mode,
        "feature_cols_used": feature_cols_used,
    }


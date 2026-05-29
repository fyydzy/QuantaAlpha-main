"""LLM feedback generation for forecast_search.

Reads solution_leaderboard.csv (and optionally top-k summary jsons) and asks LLM
to write a short conclusion: best solution, why, and next step suggestions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantaalpha.llm.client import APIBackend
from quantaalpha.log import logger


@dataclass(frozen=True)
class FeedbackConfig:
    output_dir: str
    goal: str
    leaderboard_csv: str | None = None
    top_k: int = 3
    feedback_json: str | None = None
    feedback_md: str | None = None


def _default_paths(out_root: Path) -> dict[str, Path]:
    return {
        "leaderboard": out_root / "solution_leaderboard.csv",
        "feedback_json": out_root / "forecast_feedback.json",
        "feedback_md": out_root / "forecast_feedback.md",
    }


def _resolve_optional_path(out_root: Path, raw: str | None, default: Path) -> Path:
    """Resolve optional path from YAML.

    Rules:
    - If raw is None/empty -> default
    - If raw is absolute -> raw
    - If raw is relative:
        - First try as repo-root relative (Path.cwd()).
        - If not exists, treat it as out_root relative.
    """
    if raw is None or str(raw).strip() in ("", "null", "None"):
        return default
    p = Path(str(raw))
    if p.is_absolute():
        return p

    # If user already provided a path that is rooted at out_root (e.g. "forecast_agent_output/河北/..."),
    # do NOT join it again with out_root.
    try:
        if tuple(p.parts[: len(out_root.parts)]) == tuple(out_root.parts):
            return Path.cwd() / p
    except Exception:
        pass

    repo_rel = Path.cwd() / p
    if repo_rel.exists():
        return repo_rel
    return out_root / p


def _build_prompt(cfg: FeedbackConfig, *, leaderboard_md: str) -> tuple[str, str]:
    system_prompt = (
        "你是一个严谨的数据分析助手。"
        "你需要根据给定的特征方案排行榜，输出可执行的结论与下一步建议。"
        "输出必须为严格 JSON。"
    )

    user_prompt = f"""请基于下面的实验结果给出总结。

【研究目标】
{cfg.goal}

【方案排行榜（CSV 已转换为 Markdown）】
{leaderboard_md}

请输出严格 JSON（不要 markdown 包裹，不要额外解释文字），字段如下：
{{
  "best_solution_id": string,
  "best_solution_name": string,
  "conclusion": string,
  "effective_features": [string],
  "business_interpretation": string,
  "next_step": string
}}
"""
    return system_prompt, user_prompt


def _extract_json_object(text: str) -> str:
    # Best-effort extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def generate_forecast_search_feedback(cfg: FeedbackConfig) -> dict[str, Any]:
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = _default_paths(out_root)

    leaderboard_path = _resolve_optional_path(out_root, cfg.leaderboard_csv, paths["leaderboard"])
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"leaderboard 不存在: {leaderboard_path}")

    df = pd.read_csv(leaderboard_path)
    if len(df) == 0:
        raise ValueError("leaderboard 为空，无法生成反馈")

    top_k = max(1, int(cfg.top_k))
    df_top = df.head(top_k).copy()
    leaderboard_md = df_top.to_markdown(index=False)

    system_prompt, user_prompt = _build_prompt(cfg, leaderboard_md=leaderboard_md)

    resp = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,  # request json_object; APIBackend will extract braces
        reasoning_flag=False,
    )
    resp = _extract_json_object(resp)
    payload = json.loads(resp)

    feedback_json_path = _resolve_optional_path(out_root, cfg.feedback_json, paths["feedback_json"])
    feedback_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    feedback_md_path = _resolve_optional_path(out_root, cfg.feedback_md, paths["feedback_md"])
    md = f"""## 研究目标
{cfg.goal}

## 最优方案
- **solution_id**: {payload.get("best_solution_id")}
- **name**: {payload.get("best_solution_name")}

## 结论
{payload.get("conclusion")}

## 关键特征
{payload.get("effective_features")}

## 业务解释
{payload.get("business_interpretation")}

## 下一步
{payload.get("next_step")}
"""
    feedback_md_path.write_text(md, encoding="utf-8")

    logger.info(f"[forecast_search] feedback saved: {feedback_json_path}")
    return {
        "feedback_json": str(feedback_json_path),
        "feedback_md": str(feedback_md_path),
        "payload": payload,
    }


__all__ = [
    "FeedbackConfig",
    "generate_forecast_search_feedback",
]


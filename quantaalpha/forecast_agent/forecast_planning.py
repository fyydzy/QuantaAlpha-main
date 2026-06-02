"""LLM planning for forecast_search: propose feature-set solutions.

Uses the project's unified LLM client (`quantaalpha.llm.client.APIBackend`).
API keys are expected to be provided via environment variables / LLM_SETTINGS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantaalpha.llm.client import APIBackend
from quantaalpha.log import logger

from quantaalpha.forecast_agent.feature_registry import FEATURE_REGISTRY, feature_pool_for_prompt
from quantaalpha.forecast_agent.solution_schema import (
    ForecastSolution,
    parse_forecast_solution_library,
)


@dataclass(frozen=True)
class PlanningConfig:
    goal: str
    number_of_solutions: int = 4
    max_feature_count: int = 10
    required_features: list[str] | None = None
    max_attempts: int = 5


def _build_prompt(cfg: PlanningConfig, *, feature_pool_json: str) -> tuple[str, str]:
    required = [f for f in (cfg.required_features or []) if f]
    required_block = ""
    if required:
        required_block = (
            "\n可选约束（若非空则必须满足）：\n"
            f"- 每个方案必须包含：{required}\n"
        )

    system_prompt = (
        "你是一个严谨的特征方案设计助手。"
        "你的任务是为旬度天然气销量预测提出候选特征组合方案。"
        "请严格遵守输出格式要求。"
    )

    user_prompt = f"""你是旬度天然气销量预测的特征方案设计助手。

【研究目标】
{cfg.goal}

【可选特征池】（只能从中选择，不得使用列表外的名称）
{feature_pool_json}

请提出 {cfg.number_of_solutions} 个彼此有差异的候选特征方案。每个方案需给出：
- solution_id（如 solution_001）
- name（简短中文名）
- hypothesis（为什么这组特征可能改善上述目标）
- feature_set（从特征池中选取的列名数组）

硬性约束：
1. feature_set 中的每个名称必须出现在【可选特征池】中；
2. 每个方案特征数量不超过 {cfg.max_feature_count}；
3. 不得发明新特征名、不得写 Python/公式、不得修改特征定义；
4. 输出为严格 JSON 数组，不要 markdown 包裹，不要多余解释文本。
{required_block}
"""
    return system_prompt, user_prompt


def propose_solutions(
    *,
    goal: str,
    number_of_solutions: int = 4,
    max_feature_count: int = 10,
    required_features: list[str] | None = None,
    feature_registry: dict[str, Any] | None = None,
    max_attempts: int = 5,
) -> list[ForecastSolution]:
    """Call LLM to propose candidate solutions.

    Returns a list of `ForecastSolution` objects validated by `solution_schema`.
    """
    if not str(goal).strip():
        raise ValueError("goal 不能为空")
    if number_of_solutions <= 0:
        raise ValueError("number_of_solutions 必须为正整数")
    if max_feature_count <= 0:
        raise ValueError("max_feature_count 必须为正整数")

    reg = FEATURE_REGISTRY if feature_registry is None else feature_registry
    feature_pool_json = feature_pool_for_prompt(registry=reg)  # stable JSON string

    cfg = PlanningConfig(
        goal=goal,
        number_of_solutions=int(number_of_solutions),
        max_feature_count=int(max_feature_count),
        required_features=required_features,
        max_attempts=int(max_attempts),
    )
    system_prompt, user_prompt = _build_prompt(cfg, feature_pool_json=feature_pool_json)

    last_exc: Exception | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,  # we need JSON array (not json_object)
                reasoning_flag=False,  # do NOT use reasoning_model (often unset on compatible endpoints)
            )
            library = parse_forecast_solution_library(resp)
            solutions = library.solutions[: cfg.number_of_solutions]
            if not solutions:
                raise ValueError("LLM 输出解析后 solutions 为空")
            return solutions
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(f"forecast_planning parse/call failed (attempt {attempt}): {exc}")
            # Strengthen prompt on retry.
            system_prompt = system_prompt + "\n\n你必须只输出严格 JSON 数组，不得输出任何多余文本。"

    raise RuntimeError(f"LLM 方案生成失败（已重试 {cfg.max_attempts} 次）: {last_exc}")


__all__ = [
    "PlanningConfig",
    "propose_solutions",
]


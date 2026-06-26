"""Forecast flow LLM wrapper with lightweight retry."""

from __future__ import annotations

import os
import time

from quantaalpha.llm.client import APIBackend
from quantaalpha.log import logger


def _retry_times() -> int:
    raw = str(os.getenv("FORECAST_LLM_RETRY_TIMES", "2")).strip()
    try:
        val = int(raw)
    except Exception:
        val = 2
    return max(0, val)


def _retry_sleep_ms() -> int:
    raw = str(os.getenv("FORECAST_LLM_RETRY_SLEEP_MS", "400")).strip()
    try:
        val = int(raw)
    except Exception:
        val = 400
    return max(0, val)


def create_chat_completion_with_retry(
    *,
    user_prompt: str,
    system_prompt: str,
    json_mode: bool = True,
    reasoning_flag: bool = False,
    chat_api_key: str | None = None,
    scene: str = "forecast_flow",
    **kwargs: object,
) -> str:
    """Call APIBackend with bounded retries, raising last exception on failure."""
    retries = _retry_times()
    sleep_ms = _retry_sleep_ms()
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            backend = APIBackend(chat_api_key=chat_api_key or None)
            return backend.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                reasoning_flag=reasoning_flag,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            delay = sleep_ms * (attempt + 1)
            logger.warning(
                f"{scene} LLM 调用失败，第 {attempt + 1}/{retries + 1} 次，"
                f"{delay}ms 后重试: {exc}"
            )
            if delay > 0:
                time.sleep(delay / 1000.0)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{scene} LLM 调用失败: 未知错误")


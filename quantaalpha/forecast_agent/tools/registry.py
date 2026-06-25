"""Forecast tool registry for deterministic orchestration."""

from __future__ import annotations

from typing import Any, Callable

ToolCallable = Callable[..., Any]

_FORECAST_TOOLS: dict[str, ToolCallable] = {}
_BUILTINS_REGISTERED = False


def register_forecast_tool(name: str, fn: ToolCallable) -> None:
    tool_name = str(name or "").strip()
    if not tool_name:
        raise ValueError("工具名称不能为空")
    if not callable(fn):
        raise TypeError(f"工具 {tool_name} 必须是可调用对象")
    _FORECAST_TOOLS[tool_name] = fn


def get_forecast_tool(name: str) -> ToolCallable:
    ensure_builtin_forecast_tools()
    tool_name = str(name or "").strip()
    fn = _FORECAST_TOOLS.get(tool_name)
    if fn is None:
        available = ",".join(sorted(_FORECAST_TOOLS))
        raise KeyError(f"未注册工具: {tool_name}; 已注册: [{available}]")
    return fn


def list_forecast_tools() -> list[str]:
    ensure_builtin_forecast_tools()
    return sorted(_FORECAST_TOOLS.keys())


def call_forecast_tool(name: str, /, *args: Any, **kwargs: Any) -> Any:
    fn = get_forecast_tool(name)
    return fn(*args, **kwargs)


def ensure_builtin_forecast_tools() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from quantaalpha.forecast_agent.tools.builtins import built_in_forecast_tools

    for name, fn in built_in_forecast_tools().items():
        register_forecast_tool(name, fn)
    _BUILTINS_REGISTERED = True


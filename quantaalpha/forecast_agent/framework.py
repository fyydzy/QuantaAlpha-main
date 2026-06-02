from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HyperParamsLike(Protocol):
    """超参容器需满足的最小接口。"""

    def signature(self) -> tuple[Any, ...]:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...


@dataclass
class ForecastTask:
    """旬度燃气预测任务。

    Excel 主时间列为 ``date``，值为每旬开始日（每月 1/11/21 日）。
    使用 ``as_of_month`` 指定的旬开始日及以前的数据训练，并预测 bridge + test 各旬；
    测试区间由 ``test_start`` / ``test_end`` 的旬开始日指定。
    """

    output_dir: Path = Path("forecast_agent_output")
    excel_path: Path | None = None
    province: str | None = None
    as_of_month: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    # Optional: for forecast_search feature-set experiments
    solution_id: str | None = None
    solution_name: str | None = None
    hypothesis: str | None = None
    feature_set: list[str] = field(default_factory=list)


@dataclass
class ForecastSubjects:
    params: HyperParamsLike
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastFeedback:
    score: float
    smape: float
    mape: float
    rmse: float
    mae: float
    bias: float
    aic: float | None
    success: bool
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastStep:
    evolvable_subjects: ForecastSubjects
    feedback: ForecastFeedback | None = None
    proposal_reason: str = ""


class ForecastEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: ForecastTask,
        subjects: ForecastSubjects,
    ) -> ForecastFeedback:
        raise NotImplementedError


class ForecastStrategy(ABC):
    @abstractmethod
    def seed_subjects(self, task: ForecastTask) -> list[ForecastSubjects]:
        raise NotImplementedError


class ForecastAgent(ABC):
    def __init__(
        self,
        strategy: ForecastStrategy,
        evaluator: ForecastEvaluator,
    ) -> None:
        self.strategy = strategy
        self.evaluator = evaluator
        self.trace: list[ForecastStep] = []

    @abstractmethod
    def run(self, task: ForecastTask) -> dict[str, Any]:
        raise NotImplementedError

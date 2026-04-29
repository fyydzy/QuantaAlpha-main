from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ForecastTask:
    csv_path: Path
    ds_col: str = "ds"
    y_col: str = "y"
    horizon: int = 100
    validation_days: int = 180
    output_dir: Path = Path("forecast_agent_output")


@dataclass(frozen=True)
class SarimaxHyperParams:
    p: int
    d: int
    q: int
    weekly_order: int = 1
    yearly_order: int = 3
    trend: str = "c"

    def signature(self) -> tuple[int, int, int, int, int, str]:
        return (self.p, self.d, self.q, self.weekly_order, self.yearly_order, self.trend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p": self.p,
            "d": self.d,
            "q": self.q,
            "weekly_order": self.weekly_order,
            "yearly_order": self.yearly_order,
            "trend": self.trend,
        }


@dataclass
class ForecastSubjects:
    params: SarimaxHyperParams
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastFeedback:
    score: float
    smape: float
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

    @abstractmethod
    def evolve(
        self,
        best_subjects: ForecastSubjects,
        evolving_trace: list[ForecastStep],
        top_k: int = 4,
    ) -> list[ForecastSubjects]:
        raise NotImplementedError


class ForecastAgent(ABC):
    def __init__(
        self,
        max_loops: int,
        evolving_strategy: ForecastStrategy,
        evaluator: ForecastEvaluator,
    ) -> None:
        self.max_loops = max_loops
        self.evolving_strategy = evolving_strategy
        self.evaluator = evaluator
        self.evolving_trace: list[ForecastStep] = []

    @abstractmethod
    def run(self, task: ForecastTask) -> dict[str, Any]:
        raise NotImplementedError

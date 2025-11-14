from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping

from models.ParamSpace import ParamSpace


class Optimizer(ABC):
    def __init__(
        self,
        param_space: Mapping[str, ParamSpace],
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        metric_key: str = "accuracy",
        seed: int | None = None,
    ) -> None:
        self.param_space = dict(param_space)
        self.evaluate_fn = evaluate_fn
        self.metric_key = metric_key
        self.seed = seed

    @abstractmethod
    def run(self, trials: int, verbose: bool = False):
        raise NotImplementedError

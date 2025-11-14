from __future__ import annotations
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional
from torch.utils.tensorboard import SummaryWriter
from models.ParamSpace import ParamSpace, ParamType
from .base import Optimizer


# Maybe change this at some point depending on the needs
@dataclass
class RandomSearchResult:
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    trials: int
    history: List[Dict[str, Any]]


class RandomSearch(Optimizer):
    def __init__(
        self,
        param_space: Mapping[str, ParamSpace],
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        metric_key: str = "accuracy",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(param_space, evaluate_fn, metric_key, seed)
        self._random = random.Random(seed)

    def run(
        self,
        trials: int,
        verbose: bool = False,
        writer: Optional[SummaryWriter] = None,
    ):
        if trials <= 0:
            raise ValueError("trials must be a positive integer")

        best_params: Optional[Dict[str, Any]] = {}
        best_metrics: Optional[Dict[str, float]] = {}
        best_score: float = float("-inf")
        history: List[Dict[str, Any]] = []

        if verbose:
            print(f"Running {trials} trials...")
            print(f"Optimizing for metric: {self.metric_key}")
        for trial in range(1, trials + 1):
            params = self._sample_parameters()
            if verbose:
                print(f"Trial {trial}/{trials}: {params}")
            start = time.perf_counter()
            metrics = self.evaluate_fn(params)
            duration = time.perf_counter() - start
            if self.metric_key not in metrics:
                raise KeyError(
                    f"Metric '{self.metric_key}' missing from evaluation result: {metrics}"
                )
            score = metrics[self.metric_key]
            history.append(
                {
                    "trial": trial,
                    "params": params,
                    "metrics": metrics,
                    "score": score,
                    "duration_sec": duration,
                }
            )
            if writer is not None:
                writer.add_scalar("search/duration_sec", duration, trial)
                writer.add_scalar("search/score", score, trial)
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"metrics/{metric_name}", value, trial)
                writer.add_text("params/json", json.dumps(params, default=str), trial)
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
                if verbose:
                    print(f"  -> New best! {self.metric_key}={score:.4f}")

        return RandomSearchResult(
            best_params=best_params,
            best_metrics=best_metrics,
            trials=trials,
            history=history,
        )

    def _sample_parameters(self) -> Dict[str, Any]:
        sampled: Dict[str, Any] = {}
        for name, space in self.param_space.items():
            if space.param_type == ParamType.INTEGER:
                sampled[name] = self._random.randint(
                    int(space.min_value), int(space.max_value)
                )
            elif space.param_type == ParamType.FLOAT:
                sampled[name] = self._random.uniform(
                    float(space.min_value), float(space.max_value)
                )
            elif space.param_type == ParamType.CATEGORICAL:
                sampled[name] = self._random.choice(space.choices)
            elif space.param_type == ParamType.BOOLEAN:
                sampled[name] = self._random.choice(space.choices)
            else:
                raise ValueError(
                    f"Unsupported parameter type for '{name}': {space.param_type}"
                )
        return sampled

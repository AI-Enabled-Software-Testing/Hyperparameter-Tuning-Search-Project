from __future__ import annotations

from .base import Optimizer


class GeneticAlgorithm(Optimizer):
    def run(self, trials: int, verbose: bool = False):
        raise NotImplementedError()

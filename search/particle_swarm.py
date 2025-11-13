from __future__ import annotations


from .base import Optimizer


class ParticleSwarmOptimization(Optimizer):
    def run(self, trials: int, verbose: bool = False):
        raise NotImplementedError()

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional
import numpy as np
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter
from models.ParamSpace import ParamSpace, ParamType
from .base import Optimizer


class ParameterTransformer:
    """Transforms the parameter space into a vector and back."""

    def __init__(self, param_space: Mapping[str, ParamSpace]):
        self.param_space = param_space
        self.param_names = sorted(param_space.keys())
        
        # Build layout
        self.dim_slices: Dict[str, slice] = {}
        self.total_dim = 0
        self.bounds_min: List[float] = []
        self.bounds_max: List[float] = []
        
        self.types: List[ParamType] = [] 

        for name in self.param_names:
            space = param_space[name]
            start_idx = self.total_dim
            
            if space.param_type in [ParamType.INTEGER, ParamType.FLOAT]:
                # 1 Dimension, linear bounds
                self.total_dim += 1
                self.bounds_min.append(float(space.min_value))
                self.bounds_max.append(float(space.max_value))
                self.types.append(space.param_type)
                
            elif space.param_type == ParamType.FLOAT_LOG:
                # 1 Dimension, Logarithmic bounds
                self.total_dim += 1
                self.bounds_min.append(math.log(float(space.min_value)))
                self.bounds_max.append(math.log(float(space.max_value)))
                self.types.append(space.param_type)
                
            elif space.param_type in [ParamType.CATEGORICAL, ParamType.BOOLEAN]:
                # One-hot ndim
                choices = space.choices
                if choices is None:
                    raise ValueError(f"choices cannot be None for {space.param_type.value} parameter")
                n_choices = len(choices)
                
                self.total_dim += n_choices
                # Clamp to prevent saturation
                self.bounds_min.extend([-10.0] * n_choices)
                self.bounds_max.extend([10.0] * n_choices)
                self.types.extend([space.param_type] * n_choices)
            
            self.dim_slices[name] = slice(start_idx, self.total_dim)

        self.np_bounds_min = np.array(self.bounds_min, dtype=float)
        self.np_bounds_max = np.array(self.bounds_max, dtype=float)
        
        # Velocity limits: 20% of the range
        # kinda arbitrary but it works.
        self.vel_limits = (self.np_bounds_max - self.np_bounds_min) * 0.2

    def vector_to_params(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert a PSO vector back to a dictionary for the model."""
        params = {}
        
        for name in self.param_names:
            space = self.param_space[name]
            sl = self.dim_slices[name]
            segment = vector[sl]
            
            if space.param_type == ParamType.INTEGER:
                if space.min_value is None or space.max_value is None:
                    raise ValueError("min_value and max_value required for INTEGER parameter")
                rounded = int(round(float(segment[0])))
                params[name] = max(int(space.min_value), min(int(space.max_value), rounded))
                
            elif space.param_type == ParamType.FLOAT:
                if space.min_value is None or space.max_value is None:
                    raise ValueError("min_value and max_value required for FLOAT parameter")
                val = float(segment[0])
                params[name] = float(max(float(space.min_value), min(float(space.max_value), val)))
                
            elif space.param_type == ParamType.FLOAT_LOG:
                if space.min_value is None or space.max_value is None:
                    raise ValueError("min_value and max_value required for FLOAT_LOG parameter")
                exp_val = math.exp(float(segment[0]))
                params[name] = float(max(float(space.min_value), min(float(space.max_value), exp_val)))
                
            elif space.param_type in [ParamType.CATEGORICAL, ParamType.BOOLEAN]:
                if space.choices is None:
                    raise ValueError(f"choices cannot be None for {space.param_type.value} parameter")
                best_idx = np.argmax(segment)
                params[name] = space.choices[best_idx]
                
        return params

    def sample_random_vector(self, rng: random.Random) -> np.ndarray:
        """Sample a random valid vector in the search space."""
        vec = np.zeros(self.total_dim)
        
        # I'm choosing a random value between -2 and 2 for the one-hot ndim.
        for i, (b_min, b_max, p_type) in enumerate(zip(self.bounds_min, self.bounds_max, self.types)):
            if p_type in [ParamType.CATEGORICAL, ParamType.BOOLEAN]:
                vec[i] = rng.uniform(-2.0, 2.0)
            else:
                vec[i] = rng.uniform(b_min, b_max)
        return vec


@dataclass
class PSOResult:
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    trials: int
    history: List[Dict[str, Any]]


class _Particle:
    def __init__(
        self,
        transformer: ParameterTransformer,
        rng: random.Random
    ) -> None:
        self.transformer = transformer
        self.position = transformer.sample_random_vector(rng)
        self.velocity = np.zeros_like(self.position)
        self.p_best_pos = self.position.copy()
        self.p_best_score = float("-inf")
        self.current_params_dict = transformer.vector_to_params(self.position)

    def update_velocity(
        self,
        w: float,
        c1: float,
        c2: float,
        r1: np.ndarray,
        r2: np.ndarray,
        g_best_pos: np.ndarray
    ) -> None:
        # Standard PSO
        # v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        
        cognitive = c1 * r1 * (self.p_best_pos - self.position)
        social = c2 * r2 * (g_best_pos - self.position)
        
        self.velocity = (w * self.velocity) + cognitive + social
        
        # Clip velocity to prevent explosion
        self.velocity = np.clip(
            self.velocity, 
            -self.transformer.vel_limits, 
            self.transformer.vel_limits
        )

    def move(self) -> None:
        self.position += self.velocity
        
        # Clamp position to valid bounds
        self.position = np.clip(
            self.position, 
            self.transformer.np_bounds_min, 
            self.transformer.np_bounds_max
        )
        
        self.current_params_dict = self.transformer.vector_to_params(self.position)


class ParticleSwarmOptimization(Optimizer):
    def __init__(
        self,
        param_space: Mapping[str, ParamSpace],
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        metric_key: str = "accuracy",
        seed: Optional[int] = None,
        n_jobs: int | None = 1,
        # PSO Hyperparams
        n_particles: int = 10,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> None:
        super().__init__(param_space, evaluate_fn, metric_key, seed)
        # Vector to param space transformer
        self.transformer = ParameterTransformer(self.param_space)

    def run(
        self,
        trials: int,
        verbose: bool = False,
        writer: Optional[SummaryWriter] = None,
    ) -> PSOResult:
        if trials <= 0:
            raise ValueError("trials must be positive")

        if verbose:
            print(f"Starting PSO: {self.n_particles} particles, {trials} total budget")
            print(f"Dimensions: {self.transformer.total_dim} (Logit Space)")
            if self.n_jobs != 1:
                if self.n_jobs == -1:
                    print("Using all available CPUs for parallel execution")
                else:
                    print(f"Using {self.n_jobs} parallel workers")

        history: List[Dict[str, Any]] = []
        
        g_best_pos: Optional[np.ndarray] = None
        g_best_score = float("-inf")
        g_best_metrics: Dict[str, float] = {}
        g_best_params: Dict[str, Any] = {}

        swarm = [
            _Particle(self.transformer, self._rng) 
            for _ in range(self.n_particles)
        ]

        evals_done = 0
        generation = 0

        while evals_done < trials:
            generation += 1
            
            # Update Kinematics (Skip gen 0)
            if evals_done > 0 and g_best_pos is not None:
                for p in swarm:
                    # Random vectors for stochasticity
                    r1 = np.random.rand(self.transformer.total_dim)
                    r2 = np.random.rand(self.transformer.total_dim)
                    
                    p.update_velocity(self.w, self.c1, self.c2, r1, r2, g_best_pos)
                    p.move()

            remaining = trials - evals_done
            current_batch = swarm[:remaining]
            configs = [p.current_params_dict for p in current_batch]

            if self.n_jobs == 1:
                results = []
                for cfg in configs:
                    t0 = time.perf_counter()
                    m = self.evaluate_fn(cfg)
                    dt = time.perf_counter() - t0
                    results.append((m, dt))
            else:
                def _eval_wrapper(c):
                    t0 = time.perf_counter()
                    m = self.evaluate_fn(c)
                    dt = time.perf_counter() - t0
                    return m, dt
                
                parallel_v = 10 if verbose else 0
                results = Parallel(n_jobs=self.n_jobs, verbose=parallel_v)(
                    delayed(_eval_wrapper)(c) for c in configs
                )

            # Update Knowledge
            for i, (metrics, duration) in enumerate(results):
                p = current_batch[i]
                evals_done += 1
                
                score = metrics.get(self.metric_key, float("-inf"))
                
                # Update personal bests
                if score > p.p_best_score:
                    p.p_best_score = score
                    p.p_best_pos = p.position.copy()

                # Update global bests
                if score > g_best_score:
                    g_best_score = score
                    g_best_pos = p.position.copy()
                    g_best_params = p.current_params_dict.copy()
                    g_best_metrics = metrics
                    
                    if verbose:
                        print(f"  Gen {generation}: New Best {self.metric_key}={score:.4f}")

                rec = {
                    "trial": evals_done,
                    "params": p.current_params_dict.copy(),
                    "metrics": metrics,
                    "score": score,
                    "duration_sec": duration,
                }
                history.append(rec)
                
                if writer:
                    writer.add_scalar("search/score", score, evals_done)
                    writer.add_text("params", json.dumps(rec["params"], default=str), evals_done)

            if evals_done >= trials:
                break

        # Sort history by trial number
        history.sort(key=lambda x: x["trial"])

        return PSOResult(
            best_params=g_best_params,
            best_metrics=g_best_metrics,
            trials=evals_done,
            history=history,
        )
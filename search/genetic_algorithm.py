from __future__ import annotations
from typing import Any, Callable, Dict, List, Mapping, Optional

from models.ParamSpace import ParamSpace, ParamType

from .base import Optimizer

import random
import math
from joblib import Parallel, delayed

# Logging
from torch.utils.tensorboard import SummaryWriter
import time
import json

class SearchResult:
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    trials: int
    history: List[Dict[str, Any]]

# Referenced from: https://ieeexplore.ieee.org/document/8516989
class GeneticAlgorithm(Optimizer):

    def __init__(
            self, 
            param_space: Mapping[str, ParamSpace],
            evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
            metric_key: str = "accuracy",
            seed: Optional[int] = None,
            n_jobs: int | None = 1, # for Parallel Runs
        ):
        super().__init__(param_space, evaluate_fn, metric_key, seed)
        self.n_jobs = n_jobs if n_jobs is not None else -1
        # Set random seed
        self._random = random.Random(seed)


    def run(self, trials: int, populationSize: int, generations: int, radius: int | None, verbose: bool = False, writer: Optional[SummaryWriter] = None):
        """Run the Genetic Algorithm optimization process."""
        if trials <= 0:
            raise ValueError("trials must be a positive integer")
        if populationSize <= 0:
            raise ValueError("populationSize must be a positive integer")
        if generations <= 0:
            raise ValueError("generations must be a positive integer")
        if radius is not None and radius <= 0:
            raise ValueError("radius must be a positive integer if provided")
        
        # Verbose is for Logging
        if verbose:
            print(f"Running {trials} trials...")
            print(f"Optimizing for metric: {self.metric_key}")
            if self.n_jobs != 1:
                if self.n_jobs == -1:
                    print("Using all available CPUs for parallel execution")
                else:
                    print(f"Using {self.n_jobs} parallel workers")
        
        all_params = self._initialize_population(populationSize) # Initial Population
        results = []
        for gen in range(generations):
            if verbose:
                print(f"Generation {gen+1}/{generations}")
            
            if self.n_jobs == 1:
                for geneID, params in enumerate(all_params, start=1):
                    # all_params is a List of Dicts
                    # params is a Dict of parameter name => value (i.e., one valid combination)
                    if verbose:
                        print(f"Population {geneID}/{populationSize}: {params}")
                    # Evaluate the fitness of each member in the initial population
                    start = time.perf_counter()
                    metrics = self.evaluate_fn(params)
                    duration = time.perf_counter() - start
                    results.append((geneID, params, metrics, duration))
            else: # Parallel Processing
                def evaluate_population(geneID, params):
                    """Evaluate a single population member."""
                    start = time.perf_counter()
                    metrics = self.evaluate_fn(params)
                    duration = time.perf_counter() - start
                    return (geneID, params, metrics, duration)
                
                parallel_verbose = 10 if verbose else 0
                # Record results (mainly the individual fitness values) into an iterable structure
                results += list(Parallel(n_jobs=self.n_jobs, verbose=parallel_verbose)(
                    delayed(evaluate_population)(geneID, params)
                    for geneID, params in enumerate(all_params, start=1) # each gene and param value in a parameters set
                ))

            ###################################
            # Apply Genetic Operators: Selection, Crossover, Mutation here
            # 0. Data Preparation
            # Convert generator to list if needed
            results = list(results)  # Now safe to index
            fitness_scores = [res[2][self.metric_key] for res in results]

            # Optional: Memetic to escape Local Search
            if radius is not None:
                if verbose:
                    print(f"It is a Memetic GA with local search radius: {radius}")
                # Select fittest individuals for local search
                all_params = self._selection(
                    all_params,
                    fitness_scores,
                    radius # Get the fittest S individuals from the population
                )

            # 1. Selection
            elites: List[Dict[str, Any]] = self._selection(
                all_params, # Initial Population
                fitness_scores, # Get the Fitness scores from the 3rd element
                populationSize // 2 # Elitism: shortlisting top 50%
            )
            # A brief Checking
            assert len(elites) < len(all_params), "Chosen Elites should be fewer than the total population"
            assert len(elites) < populationSize, "Number of selected Elites should be less than the population size"

            # 2. Crossover
            offspring: List[Dict[str, Any]] = [] # They are evolved children
            # Choose parents for crossover from elites
            while len(offspring) < (populationSize - len(elites)):
                parent1 = self._random.choice(elites)
                parent2 = self._random.choice(elites)
                # Ensure parents are not identical
                if parent1 != parent2:
                    # The papar suggests 1-point crossover
                    child = self._crossover(parent1, parent2, n=1)
                    offspring.append(child)
            # A Brief Checking
            assert len(offspring) < populationSize, "Number of Offsprings should be less than the population size"
            assert len(offspring) < len(all_params), "Evolved Offsprings should be fewer than the total population"
            
            # 3. Mutation
            # We get a list of mutated children from crossover offsprings
            mutated_offspring: List[Dict[str, Any]] = []
            for child in offspring:
                mutated_child = self._mutate(child)
                mutated_offspring.append(mutated_child)
            # A Brief Checking
            assert len(mutated_offspring) == len(offspring), "Number of Mutated Offsprings should match the Offsprings"

            # Replacing the population with new generation: elites + offspring + mutated_offspring
            all_params = elites + offspring + mutated_offspring
            # A Brief Checking
            assert len(all_params) >= populationSize, "New generation population size should be at least the defined population size"
            
            if verbose:
                print(f"Generation {gen+1} completed. Current Population size: {len(all_params)}")

            ###################################        

        # Process results and build history
        best_params: Optional[Dict[str, Any]] = {}
        best_metrics: Optional[Dict[str, float]] = {}
        best_score: float = float("-inf")
        history: List[Dict[str, Any]] = []

        for trial, params, metrics, duration in results:
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

        history.sort(key=lambda x: x["trial"])

        return SearchResult(
            best_params=best_params,
            best_metrics=best_metrics,
            trials=trials,
            history=history,
        )
    
    #################################
    # GA Specifics
    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample a set of parameters RANDOMLY from the parameter space."""
        sampled: Dict[str, Any] = {}
        for name, space in self.param_space.items():
            # Based on their types, sample accordingly
            if space.param_type == ParamType.INTEGER:
                sampled[name] = self._random.randint(
                    int(space.min_value), int(space.max_value)
                )
            elif space.param_type == ParamType.FLOAT:
                sampled[name] = self._random.uniform(
                    float(space.min_value), float(space.max_value)
                )
            elif space.param_type == ParamType.FLOAT_LOG:
                log_min = math.log(float(space.min_value))
                log_max = math.log(float(space.max_value))
                sampled[name] = math.exp(self._random.uniform(log_min, log_max))
            elif space.param_type == ParamType.CATEGORICAL:
                sampled[name] = self._random.choice(space.choices)
            elif space.param_type == ParamType.BOOLEAN:
                sampled[name] = self._random.choice(space.choices)
            else:
                raise ValueError(
                    f"Unsupported parameter type for '{name}': {space.param_type}"
                )
        return sampled

    def _initialize_population(self, population_size: int) -> List[Dict[str, Any]]:
        """Initialize the population with random parameter sets."""
        return [self._sample_parameters() for _ in range(population_size)]
    
    def _selection(self, population: List[Dict[str, Any]], fitness_scores: List[float], num_parents: int) -> List[Dict[str, Any]]:
        """Select the top individuals based on fitness scores."""
        # Pair each individual with its fitness score
        paired = list(zip(population, fitness_scores))
        # Sort by fitness score in descending order
        paired.sort(key=lambda x: x[1], reverse=True) # Only 2 Genes in a pair
        # Select the top individuals - top num_parents
        # Only return the individuals, not their scores
        selected = [individual for individual, score in paired[:num_parents]]
        return selected

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], n: int = 1) -> Dict[str, Any]:
        """Perform n-point crossover between two parent genes."""
        child: Dict[str, Any] = {}
        # Generate n unique crossover points
        if n <= 0:
            raise ValueError("Number of crossover points 'n' must be a positive integer")
        # len(parent) is the Number of Paramaters in a set for a certain model
        # We got n-crossover points of chosen indexes of the parameters to swap
        crossover_points = sorted(
            self._random.sample( # random gene section from a parent 
                range(1, len(parent1)), 
                # Random Sample n sections of a gene (Without Replacement)
                min(
                    n, 
                    len(parent1) - 1 # Ensure n doesn't exceed parameter count
                )
            )
        )
        
        # Swapping - to produce a newly unseen solution
        for crossover_point in crossover_points: # adapt to n-point crossover
            for paramInd, key in enumerate(parent1.keys()): # key is the parameter name in string format
                if paramInd < crossover_point:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
        return child

    def _mutate(self, individual: Dict[str, Any], mutation_rate: float = 0.01, mutation_strength: float = 0.1) -> Dict[str, Any]:
        """Mutate an individual with a given mutation rate."""
        mutated_individual = individual.copy()
        
        for key, current_value in mutated_individual.items():
            chance = self._random.random()
            if chance < mutation_rate:
                # Alter a Random Gene in the Individual in the given fraction p_m of children
                space = self.param_space[key] # Keep the same parameter space
                
                # Apply mutation based on parameter type
                if space.param_type == ParamType.INTEGER:
                    # Small integer perturbation within bounds
                    range_size = int(space.max_value) - int(space.min_value)
                    max_delta = max(1, int(range_size * mutation_strength))
                    delta = self._random.randint(-max_delta, max_delta)
                    # Keep mutated values within valid parameter bounds
                    new_value = max( # Clamping to make sure within bounds
                        int(space.min_value), 
                        min(
                            int(space.max_value), 
                            current_value + delta
                        )
                    )
                    # Update the mutated individual
                    mutated_individual[key] = new_value
                    
                elif space.param_type == ParamType.FLOAT:
                    # Gaussian (float-based) perturbation for continuous values
                    range_size = float(space.max_value) - float(space.min_value)
                    std_dev = range_size * mutation_strength
                    delta = self._random.gauss(0, std_dev)
                    # Keep mutated values within valid parameter bounds
                    new_value = max( # Clamping to make sure within bounds
                        float(space.min_value),
                        min(
                            float(space.max_value), 
                            current_value + delta
                        )
                    )
                    # Update the mutated individual
                    mutated_individual[key] = new_value
                    
                elif space.param_type == ParamType.FLOAT_LOG:
                    # Log-scale perturbation (for things like learning rates)
                    log_current = math.log(current_value)
                    log_min = math.log(float(space.min_value))
                    log_max = math.log(float(space.max_value))
                    log_range = log_max - log_min
                    std_dev = log_range * mutation_strength
                    delta = self._random.gauss(0, std_dev) # Also Gaussian perturbation
                    # Keep mutated values within valid parameter bounds with a delta
                    new_log_value = max(log_min, min(log_max, log_current + delta))
                    # Update the mutated individual
                    mutated_individual[key] = math.exp(new_log_value)
                    
                elif space.param_type in [ParamType.CATEGORICAL, ParamType.BOOLEAN]:
                    # For categorical/boolean: random choice from available options
                    # (This is appropriate since there's no "nearby" concept for discrete choices)
                    mutated_individual[key] = self._random.choice(space.choices)
                    
        return mutated_individual
from __future__ import annotations
from typing import Any, Callable, Dict, List, Mapping, Optional

from models.ParamSpace import ParamSpace, ParamType

from .base import Optimizer

import random
import math
from joblib import Parallel, delayed
from dataclasses import dataclass
from multiprocessing import Manager

# Logging
from torch.utils.tensorboard import SummaryWriter
import time
import json

@dataclass
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
            # GA Specific Parameters, default suggested by the paper
            populationSize: int = 30,
            generations: int = 10, 
            radius: float = 0.15, # for Memetic GA
            num_fittest: int | None = 5
        ):
        super().__init__(param_space, evaluate_fn, metric_key, seed)
        self.n_jobs = n_jobs if n_jobs is not None else -1
        # Set random seed
        self._random = random.Random(seed)
        # Set GA specific parameters
        if populationSize <= 0:
            raise ValueError("populationSize must be a positive integer")
        if generations <= 0:
            raise ValueError("generations must be a positive integer")
        if radius is not None and radius <= 0:
            raise ValueError("radius must be a positive number if provided")
        if num_fittest is not None and num_fittest <= 0:
            raise ValueError("num_fittest must be a positive integer if provided")
        
        self.populationSize = populationSize
        self.generations = generations
        self.radius = radius
        self.num_fittest = num_fittest
        # Memoization cache: maps trial number to (params, metrics, duration)
        # Use Manager().dict() for thread-safe access in parallel execution
        self._manager = Manager()
        self._eval_cache = self._manager.dict()

    def run(self, trials: int, verbose: bool = False, writer: Optional[SummaryWriter] = None):
        """Run the Genetic Algorithm optimization process."""
        if trials <= 0:
            raise ValueError("trials must be a positive integer")
        
        # Verbose is for Logging
        if verbose:
            print(f"Running {trials} trials...")
            print(f"Optimizing for metric: {self.metric_key}")
            if self.n_jobs != 1:
                if self.n_jobs == -1:
                    print("Using all available CPUs for parallel execution")
                else:
                    print(f"Using {self.n_jobs} parallel workers")
        
        all_params = self._initialize_population(self.populationSize) # Initial Population
        assert len(all_params) == self.populationSize, "Initial population size should match the defined population size"
        results = []
        evals_done = 0 # To control the budget with `trials`

        for gen in range(self.generations+1):
            if evals_done >= trials:
                if verbose:
                    print(f"Reached the maximum number of trials: {trials}. Stopping early at generation {gen+1} out of {self.generations}.")
                break
            if verbose:
                print(f"Generation {gen+1}/{self.generations}")

            # Only evaluate up to the remaining budget
            remaining = trials - evals_done
            all_params = all_params[:remaining] if remaining < len(all_params) else all_params

            if self.n_jobs == 1:
                for geneID, params in enumerate(all_params, start=1):
                    # all_params is a List of Dicts
                    # params is a Dict of parameter name => value (i.e., one valid combination)
                    if verbose:
                        print(f"Population {geneID}/{self.populationSize}: {params}")
                    
                    # Check memoization cache to avoid re-evaluation based on trial number
                    cache_key = evals_done
                    if cache_key in self._eval_cache:
                        # Use cached result
                        cached_params, metrics, duration = self._eval_cache[cache_key]
                        if verbose:
                            print(f"  -> Using cached evaluation result for trial {evals_done}")
                    else:
                        # Evaluate the fitness of each member in the initial population
                        start = time.perf_counter()
                        metrics = self.evaluate_fn(params)
                        duration = time.perf_counter() - start
                        # Store in cache by trial number (thread-safe with Manager.dict())
                        self._eval_cache[cache_key] = (params, metrics, duration)
                    
                    # trial, params, metrics, duration
                    if verbose:
                        print(f"Current Trial has evaluated models {evals_done} times, with: {self.metric_key} = {metrics.get(self.metric_key, 'N/A')}, Duration: {duration:.4f} sec")
                    results.append((evals_done, params, metrics, duration))
            else: # Parallel Processing
                def evaluate_population(trial_id, params, eval_cache, evaluate_fn, metric_key, verbose_flag):
                    """Evaluate a single population member (thread-safe)."""
                    # Check memoization cache to avoid re-evaluation based on trial number
                    cache_key = trial_id
                    if cache_key in eval_cache:
                        # Use cached result (thread-safe read from Manager.dict())
                        cached_params, metrics, duration = eval_cache[cache_key]
                        if verbose_flag:
                            print(f"  -> Using cached evaluation result for trial {trial_id}")
                    else:
                        start = time.perf_counter()
                        metrics = evaluate_fn(params)
                        duration = time.perf_counter() - start
                        # Store in cache by trial number (thread-safe write with Manager.dict())
                        eval_cache[cache_key] = (params, metrics, duration)
                    
                    # trial, params, metrics, duration
                    if verbose_flag:
                        print(f"Current Trial has evaluated models {trial_id} times, with: {metric_key} = {metrics.get(metric_key, 'N/A')}, Duration: {duration:.4f} sec")
                    return (trial_id, params, metrics, duration)
                
                parallel_verbose = 10 if verbose else 0
                # Record results (mainly the individual fitness values) into an iterable structure
                results += list(Parallel(n_jobs=self.n_jobs, verbose=parallel_verbose)(
                    delayed(evaluate_population)(
                        evals_done + idx,  # trial_id for uniqueness
                        params, 
                        self._eval_cache,  # Thread-safe Manager.dict()
                        self.evaluate_fn,  # Pass function to avoid closure issues
                        self.metric_key,   # Pass metric_key
                        verbose            # Pass verbose flag
                    )
                    for idx, params in enumerate(all_params) # each gene and param value in a parameters set
                ))
            
            # After Evaluation, Update the number of evaluations done
            evals_done += len(all_params)

            ###################################
            # Apply Genetic Operators: Selection, Crossover, Mutation here
            # 0. Data Preparation
            # Convert generator to list if needed
            results = list(results)  # Now safe to index
            fitness_scores = [res[2][self.metric_key] for res in results]
            elites = all_params.copy()

            # Optional: Memetic to escape Local Search
            if self.radius is not None:
                if verbose:
                    print(f"It is a Memetic GA with local search radius: {self.radius}")
                # Select fittest individuals for local search
                elites = self._selection(
                    all_params,
                    fitness_scores,
                    radius=self.radius, # Get the fittest r% individuals according to radius
                    num_parents=None # Placeholder
                )
            else:
                if verbose:
                    print("Standard Genetic Algorithm runs without local search.")

            # 1. Selection
            elites: List[Dict[str, Any]] = self._selection(
                elites, # Initial Population
                fitness_scores, # Get the Fitness scores from the 3rd element
                radius=None, # Placeholder
                # Use the current length of all_params because it might be shortlisted if MA
                num_parents=len(all_params) // 2 # Elitism: shortlisting top 50%, also suggested by the paper
            )
            # A brief Checking
            if len(elites) > len(all_params):
                if verbose:
                    print("Number of Elites exceeded total population; trimming elites.")
                elites = elites[:len(all_params)]
            if len(elites) > self.populationSize:
                if verbose:
                    print("Number of Elites exceeded population size; trimming elites.")
                elites = elites[:self.populationSize]

            # 2. Crossover
            offspring: List[Dict[str, Any]] = [] # They are evolved children
            max_offspring = max(0, self.populationSize - len(elites))
            attempts = 0
            max_attempts = max_offspring * 10  # Prevent infinite loop
            while len(offspring) < max_offspring and attempts < max_attempts:
                # Error Handling
                if len(elites) == 0:
                    # Fallback to entire population if no elites
                    elites = all_params
                    if verbose:
                        print("No elites selected; reverting to entire population for parent selection.")
                if len(offspring) > max_offspring:
                    if verbose:
                        print("Reached maximum number of offsprings allowed.")
                    break
                if len(offspring) > len(all_params):
                    if verbose:
                        print("Number of offsprings exceeded total population; stopping crossover.")
                    break
                parent1 = self._random.choice(elites)
                parent2 = self._random.choice(elites)
                # Ensure parents are not identical
                if parent1 != parent2:
                    # The papar suggests 1-point crossover
                    child = self._crossover(parent1, parent2, n=1)
                    offspring.append(child)
                attempts += 1

            # 3. Mutation
            # We get a list of mutated children from crossover offsprings
            mutated_offspring: List[Dict[str, Any]] = []
            for child in offspring:
                mutated_child = self._mutate(child)
                mutated_offspring.append(mutated_child)
            # A Brief Checking
            if len(mutated_offspring) != len(offspring):
                if verbose:
                    print("Number of Mutated Offsprings does not match the Offsprings; adjusting accordingly.")
                mutated_offspring = random.sample(mutated_offspring, len(offspring))

            # Replacing the population with new generation: elites + mutated_offspring
            new_generation = elites + mutated_offspring
            # If new_generation is less than populationSize, fill with random samples
            if len(new_generation) < self.populationSize:
                needed = self.populationSize - len(new_generation)
                new_generation += self._initialize_population(needed)
            # If new_generation is more than populationSize, trim it
            if len(new_generation) > self.populationSize:
                new_generation = new_generation[:self.populationSize]
            all_params = new_generation
            if len(all_params) != self.populationSize:
                if verbose:
                    print("Adjusting new generation to match population size.")
                all_params = random.sample(all_params, self.populationSize)

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
            # Check duplicated trial
            if any(entry["trial"] == trial for entry in history):
                # Take the Dict with best metric scores
                if verbose:
                    print(f"  -> Duplicate trial {trial} detected, checking for better score...")
                existing_entry = next(
                    entry for entry in history if entry["trial"] == trial
                )
                if score > existing_entry["score"]:
                    if verbose:
                        print(f"     -> Updating trial {trial} with better score: {score:.4f} (was {existing_entry['score']:.4f})")
                    existing_entry.update(
                        {
                            "params": params,
                            "metrics": metrics,
                            "score": score,
                            "duration_sec": duration,
                        }
                    )
                elif score == existing_entry["score"]:
                    # tie
                    if verbose:
                        print(f"     -> Tie detected for trial {trial} with score: {score:.4f}, applying tiebreaker.")
                    # Keep the search with less duration
                    if duration < existing_entry["duration_sec"]:
                        if verbose:
                            print(f"        -> Tiebreaker won! Updating trial {trial} with shorter duration: {duration:.4f} sec (was {existing_entry['duration_sec']:.4f} sec)")
                        existing_entry.update(
                            {
                                "params": params,
                                "metrics": metrics,
                                "score": score,
                                "duration_sec": duration,
                            }
                        )
            else:
                # Normal: no duplicates
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
            elif score == best_score:
                if verbose:
                    print(f"  -> Tie detected for {self.metric_key}={score:.4f}, applying tiebreaker.")
                # In case of a tie
                if not best_params:
                    # in case if best_params is not found yet
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                    if verbose:
                        print(f"  -> New best! {self.metric_key}={score:.4f}")
                else:
                    if verbose:
                        print(f"  -> Existing best params: {best_params}")
                        print(f"  -> Current params: {params}")
                    # Keep the search with less duration
                    existing_duration = next(
                        (item["duration_sec"]
                        for item in history
                        if item["params"] == best_params),
                        float('inf')  # Default to infinity if not found
                    )
                    if duration < existing_duration:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
                        if verbose:
                            print(f"  -> Tiebreaker won! Updated best params with shorter duration.")
                            print(f"     New best params: {best_params} with duration {duration:.4f} sec")

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
    
    def _selection(self, population: List[Dict[str, Any]], fitness_scores: List[float], num_parents: int | None = None, radius: float | None = 0.15) -> List[Dict[str, Any]]:
        """Select the top individuals based on fitness scores."""
        # Pair each individual with its fitness score
        paired = list(zip(population, fitness_scores))
        # Sort by fitness score in descending order
        paired.sort(key=lambda x: x[1], reverse=True) # Only 2 Genes in a pair
        # Select the top individuals - top num_parents
        # Only return the individuals, not their scores
        selected: List[Dict[str, Any]] = []
        if num_parents is not None:
            selected = [individual for individual, score in paired[:num_parents]]
        else:
            if radius is None or radius <= 0 or radius > 1:
                raise ValueError("radius must be in the range (0, 1]")
            # Only get the top <radius> % of individuals
            cutoff = int(len(paired) * radius)
            # Paired is already sorted
            selected = [individual for individual, score in paired[:cutoff]]
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
        
        # Swapping - to produce a newly unseen solution using n-point crossover
        # Alternate between parents at each crossover point
        current_parent = 0  # Start with parent1
        next_point_idx = 0
        for paramInd, key in enumerate(parent1.keys()): # key is the parameter name in string format
            # Check if we've reached the next crossover point
            if next_point_idx < len(crossover_points) and paramInd >= crossover_points[next_point_idx]:
                current_parent = 1 - current_parent  # Switch parent
                next_point_idx += 1
            # Assign value from current parent
            child[key] = parent1[key] if current_parent == 0 else parent2[key]
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
"""Experiment runner script for hyperparameter optimization experiments."""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.decision_tree import DecisionTreeModel
from models.knn import KNNModel

import numpy as np
import torch

from framework.data_utils import prepare_dataset
from framework.fitness import calculate_composite_fitness
from models.factory import get_model_by_name
from models.cnn import CNNModel, TrainingConfig
from search import RandomSearch, GeneticAlgorithm, ParticleSwarmOptimization
from dataclasses import replace

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / ".cache" / "experiment"
EVALUATIONS_PER_RUN = 50


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def evaluate_model(
    model_key: Literal["dt", "knn", "cnn"],
    params: Dict[str, Any],
    data: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate a model with given hyperparameters."""
    model = get_model_by_name(model_key)

    if model_key in {"dt", "knn"}:
        assert isinstance(model, (DecisionTreeModel, KNNModel))
        model.create_model(**params)
        model.train(data["train_flat"], data["train_labels"])
        metrics = model.evaluate(data["val_flat"], data["val_labels"])
    elif model_key == "cnn":
        assert isinstance(model, CNNModel)
        model.create_model(**params)
        default_config = TrainingConfig()
        config = replace(
            default_config,
            learning_rate=float(params.get("learning_rate", default_config.learning_rate)),
            weight_decay=float(params.get("weight_decay", default_config.weight_decay)),
            optimizer=params.get("optimizer", default_config.optimizer),
            batch_size=int(params.get("batch_size", default_config.batch_size)),
            patience=99999999,  # Disable early stopping
        )
        model.train(
            data["train_images"],
            data["train_labels"],
            data["val_images"],
            data["val_labels"],
            config=config,
            verbose=verbose,
        )
        metrics = model.evaluate(data["val_images"], data["val_labels"])
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    # Calculate composite fitness and add it to metrics
    composite_fitness = calculate_composite_fitness(metrics)
    metrics["composite_fitness"] = composite_fitness

    return metrics


def get_optimizer(
    optimizer_name: str,
    param_space: Dict[str, Any],
    evaluate_fn,
    seed: int,
    n_jobs: int | None = 1,
):
    """Get optimizer instance by name."""
    optimizer_map = {
        "rs": RandomSearch,
        "ga-standard": GeneticAlgorithm,
        "ga-memetic": GeneticAlgorithm,
        "pso": ParticleSwarmOptimization,
    }

    optimizer_class = optimizer_map.get(optimizer_name.lower())
    if optimizer_class is None:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available: {list(optimizer_map.keys())}"
        )

    if optimizer_name.lower() in {"rs", "pso"} \
        or optimizer_name.lower().startswith("ga"):
        return optimizer_class(
            param_space=param_space,
            evaluate_fn=evaluate_fn,
            metric_key="composite_fitness",
            seed=seed,
            n_jobs=n_jobs,
        )
    else:
        return optimizer_class(
            param_space=param_space,
            evaluate_fn=evaluate_fn,
            metric_key="composite_fitness",
            seed=seed,
        )


def extract_convergence_trace(history: list) -> Dict[str, list]:
    """Extract convergence trace (best-so-far fitness at each evaluation)."""
    best_so_far = float("-inf")
    convergence = []

    for entry in history:
        score = entry["score"]
        if score > best_so_far:
            best_so_far = score
        convergence.append({
            "evaluation": entry["trial"],
            "best_fitness": best_so_far,
            "current_fitness": score,
        })

    return {
        "evaluations": [c["evaluation"] for c in convergence],
        "best_fitness": [c["best_fitness"] for c in convergence],
        "current_fitness": [c["current_fitness"] for c in convergence],
    }


def save_experiment_results(
    run_dir: Path,
    model_key: str,
    optimizer_name: str,
    run_number: int,
    result: Any,
    total_time_sec: float,
    seed: int,
):
    """Save experiment results to JSON files."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Extract convergence trace
    convergence = extract_convergence_trace(result.history)

    # Prepare summary
    summary = {
        "model": model_key,
        "optimizer": optimizer_name,
        "run_number": run_number,
        "seed": seed,
        "evaluations": result.trials,
        "total_time_sec": total_time_sec,
        "final_fitness": result.best_metrics["composite_fitness"],
        "best_params": result.best_params,
        "best_metrics": result.best_metrics,
        "convergence_trace": convergence,
    }

    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save full history
    history_path = run_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(result.history, f, indent=2, default=str)

    # Save convergence trace separately for easy plotting
    convergence_path = run_dir / "convergence.json"
    with open(convergence_path, "w") as f:
        json.dump(convergence, f, indent=2)

    print(f"Results saved to: {run_dir}")
    print(f"  Final fitness: {result.best_metrics['composite_fitness']:.4f}")
    print(f"  Total time: {total_time_sec:.2f}s")


def run_experiment(
    model_key: Literal["dt", "knn", "cnn"],
    optimizer_name: str,
    num_runs: int = 1,
    evaluations: int = EVALUATIONS_PER_RUN,
    base_seed: int = 42,
    n_jobs: int | None = 1,
):
    """Run experiment for a model-optimizer combination."""
    print("=" * 80)
    print(f"Experiment: {model_key.upper()} + {optimizer_name.upper()}")
    print(f"Runs: {num_runs}, Evaluations per run: {evaluations}")
    print("=" * 80)

    # Prepare dataset (shared across all runs)
    print("Preparing dataset...")
    data = prepare_dataset()

    # Get parameter space
    print("Preparing parameter space...")
    param_space = get_model_by_name(model_key).get_param_space()

    # Create experiment directory
    experiment_dir = EXPERIMENT_ROOT / f"{model_key}-{optimizer_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Run each repetition
    for run_num in range(1, num_runs + 1):
        print(f"\n--- Run {run_num}/{num_runs} ---")

        # Use different seed for each run
        run_seed = base_seed + run_num - 1
        set_seeds(run_seed)

        is_parallel = n_jobs != 1
        def evaluate_fn(params):
            return evaluate_model(
                model_key,
                params,
                data,
                verbose=not is_parallel,
            )
        
        optimizer = get_optimizer(
            optimizer_name, param_space, evaluate_fn, seed=run_seed, n_jobs=n_jobs
        )

        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = experiment_dir / f"run_{timestamp}"

        # Run optimization
        print(f"Running {optimizer_name.upper()} with {evaluations} evaluations...")
        start_time = time.perf_counter()
        result = optimizer.run(trials=evaluations, verbose=True)
        total_time = time.perf_counter() - start_time

        # Save results
        save_experiment_results(
            run_dir=run_dir,
            model_key=model_key,
            optimizer_name=optimizer_name,
            run_number=run_num,
            result=result,
            total_time_sec=total_time,
            seed=run_seed,
        )

    print("\n" + "=" * 80)
    print(f"Experiment complete! Results saved in: {experiment_dir}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization experiments."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["dt", "knn", "cnn"],
        help="Model to optimize (dt, knn, or cnn).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["rs", "ga-standard", "ga-memetic", "pso"],
        help="Optimizer to use (rs=RandomSearch, ga-standard=GeneticAlgorithm Standard, ga-memetic=GeneticAlgorithm Memetic (For Local Search), pso=ParticleSwarmOptimization).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs to perform (default: 1).",
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=EVALUATIONS_PER_RUN,
        help=f"Number of fitness evaluations per run (default: {EVALUATIONS_PER_RUN}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for random number generation (default: 42).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential). Use -1 for all CPUs.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle -1 for all CPUs
    n_jobs = args.n_jobs if args.n_jobs > 0 else None

    if args.optimizer is None:
        # Assume running all optimizers
        for optimizer_name in ["rs", "ga-standard", "ga-memetic", "pso"]:
            run_experiment(
                model_key=args.model,
                optimizer_name=optimizer_name,
                num_runs=args.runs,
                # The paper did specify evaluations for GA differently
                evaluations=args.evaluations if not optimizer_name.startswith("ga") else 300,
                base_seed=args.seed,
                n_jobs=n_jobs,
            )
    else:
        # Run a Specific Experiment based on optimizer's name
        run_experiment(
            model_key=args.model,
            optimizer_name=args.optimizer,
            num_runs=args.runs,
            evaluations=args.evaluations,
            base_seed=args.seed,
            n_jobs=n_jobs,
        )


if __name__ == "__main__":
    exit(main())


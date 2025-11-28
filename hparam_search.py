import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from framework.data_utils import prepare_dataset
from framework.fitness import calculate_composite_fitness
from models.cnn import CNNModel
from models.decision_tree import DecisionTreeModel
from models.factory import get_model_by_name
from models.knn import KNNModel
from search import RandomSearch

RANDOM_SEED = 321

# CNN Specific
DEFAULT_EPOCHS = 5
DEFAULT_PATIENCE = 2
REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / ".cache" / "tensorboard" / "search"


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def evaluate_model(
    model_key: Literal["dt", "knn", "cnn"],
    params: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, float]:
    model = get_model_by_name(model_key)

    if model_key in {"dt", "knn"}:
        assert isinstance(model, (DecisionTreeModel, KNNModel))
        model.create_model(**params)
        model.train(data["train_flat"], data["train_labels"])
        metrics = model.evaluate(data["val_flat"], data["val_labels"])
    elif model_key == "cnn":
        assert isinstance(model, CNNModel)
        model.create_model(**params)
        model.train(
            data["train_images"],
            data["train_labels"],
            data["val_images"],
            data["val_labels"],
        )
        metrics = model.evaluate(data["val_images"], data["val_labels"])
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
    
    metrics["composite_fitness"] = calculate_composite_fitness(metrics)
    
    return metrics


def run_search(model_key: Literal["dt", "knn", "cnn"], trials: int) -> None:
    set_seeds(RANDOM_SEED)
    print("Preparing dataset...")
    data = prepare_dataset()

    print("Preparing parameter space...")
    param_space = get_model_by_name(model_key).get_param_space()
    searcher = RandomSearch(
        param_space=param_space,
        evaluate_fn=lambda sampled: evaluate_model(model_key, sampled, data),
        metric_key="composite_fitness",
        seed=RANDOM_SEED,
    )

    log_dir = create_search_log_dir(model_key)
    print(f"Running search... (logging to {log_dir})")
    writer = SummaryWriter(log_dir=log_dir)
    try:
        result = searcher.run(trials, verbose=True, writer=writer)
    finally:
        writer.close()

    print("-" * 80)
    print(f"Model: {model_key}")
    print(f"Trials: {trials}")
    print(f"Best composite fitness: {result.best_metrics['composite_fitness']:.4f}")
    print("Best metrics:")
    for name, value in result.best_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
    print("Best hyperparameters:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value}")


def create_search_log_dir(model_key: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / model_key / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random hyperparameter search for CIFAR-10 models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dt",
        choices=["dt", "knn", "cnn"],
        help="Model to optimize.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of random search trials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_search(args.model, args.trials)


if __name__ == "__main__":
    main()

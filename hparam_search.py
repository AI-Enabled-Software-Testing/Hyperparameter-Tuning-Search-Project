import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from framework.data_utils import (
    create_dataloaders,
    load_cifar10_data,
    prepare_data,
    split_train_val,
)
from framework.datasets import CIFAR10Dataset
from framework.utils import get_device
from models.base import get_model_by_name
from models.cnn import TrainingConfig
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


def prepare_dataset() -> Dict[str, Any]:
    ds_dict = load_cifar10_data()
    train_images, train_labels = prepare_data(ds_dict, "train")
    test_images, test_labels = prepare_data(ds_dict, "test")

    X_train, y_train, X_val, y_val = split_train_val(
        train_images, train_labels, val_ratio=0.2
    )

    def flatten(images):
        stacked = np.stack([np.asarray(img, dtype=np.float32) for img in images])
        return stacked.reshape(len(images), -1)

    train_flat = flatten(X_train)
    val_flat = flatten(X_val)
    test_flat = flatten(test_images)

    return {
        "train_images": X_train,
        "train_labels": y_train,
        "val_images": X_val,
        "val_labels": y_val,
        "test_images": test_images,
        "test_labels": test_labels,
        "train_flat": train_flat,
        "val_flat": val_flat,
        "test_flat": test_flat,
    }


def evaluate_model(
    model_key: Literal["dt", "knn", "cnn"],
    params: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, float]:
    model = get_model_by_name(model_key)

    if model_key in {"dt", "knn"}:
        model.create_model(**params)
        model.train(data["train_flat"], data["train_labels"])
        return model.evaluate(data["val_flat"], data["val_labels"])

    if model_key == "cnn":
        # Architecture specific parameters
        architecture = {k: params[k] for k in ("kernel_size", "stride")}
        model.create_model(**architecture)

        # Training specific parameters
        batch_size = int(params["batch_size"])
        config = TrainingConfig(
            epochs=DEFAULT_EPOCHS,
            learning_rate=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            optimizer=params["optimizer"],
            patience=DEFAULT_PATIENCE,
            batch_size=batch_size,
        )
        train_loader, val_loader = create_dataloaders(
            data["train_images"],
            data["train_labels"],
            data["val_images"],
            data["val_labels"],
            batch_size=batch_size,
        )

        device = get_device()
        model.train(
            train_loader, val_loader, config=config, device=device
        )

        eval_loader = DataLoader(
            CIFAR10Dataset(data["val_images"], data["val_labels"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        eval_metrics = model.evaluate(eval_loader, device=device)
        return eval_metrics

    raise ValueError(f"Unsupported model key: {model_key}")


def run_search(model_key: Literal["dt", "knn", "cnn"], trials: int) -> None:
    set_seeds(RANDOM_SEED)
    print("Preparing dataset...")
    data = prepare_dataset()

    print("Preparing parameter space...")
    param_space = get_model_by_name(model_key).get_param_space()
    searcher = RandomSearch(
        param_space=param_space,
        evaluate_fn=lambda sampled: evaluate_model(model_key, sampled, data),
        metric_key="accuracy",
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
    print(f"Best val accuracy: {result.best_metrics['accuracy']:.4f}")
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

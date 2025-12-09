"""
Run final training for best-found hyperparameters and save artifacts.

- For each experiment folder under .cache/experiment/<model-optimizer>,
  load best_params from the best run's summary.json (highest final_fitness).
- Train on train set (CNNs use separate validation set for early stopping);
  evaluate on the held-out test set.
- Save model artifact and summary to .cache/final_training/<model-optimizer>/run_<timestamp>_<seed>/.
- CNN: use epochs=200 with early stopping patience=20; other models keep defaults.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.data_utils import prepare_dataset
from framework.fitness import calculate_composite_fitness
from framework.utils import get_device
from models.cnn import CNNModel, TrainingConfig as CNNTrainingConfig
from models.decision_tree import DecisionTreeModel
from models.factory import get_model_by_name
from models.knn import KNNModel

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / ".cache" / "experiment"
FINAL_ROOT = REPO_ROOT / ".cache" / "final_training"


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_best_run(exp_dir: Path) -> Dict[str, Any]:
    best = None
    best_fitness = float("-inf")
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as f:
            summary = json.load(f)
        fitness = summary.get("final_fitness") or summary.get("best_metrics", {}).get("composite_fitness")
        if fitness is None:
            continue
        if fitness > best_fitness:
            best_fitness = fitness
            best = summary
    if best is None:
        raise ValueError(f"No valid summary.json with fitness in {exp_dir}")
    return best


def train_and_eval(
    model_key: Literal["dt", "knn", "cnn"],
    params: Dict[str, Any],
    data: Dict[str, Any],
    out_dir: Path,
    device: Optional[torch.device] = None,
) -> tuple[Dict[str, Any], Optional[Path]]:
    model = get_model_by_name(model_key)
    if model_key == "dt":
        assert isinstance(model, DecisionTreeModel)
        model.create_model(**params)
        model.train(data["train_flat"], data["train_labels"])
        metrics = model.evaluate(data["test_flat"], data["test_labels"])
        artifact_path = None
    elif model_key == "knn":
        assert isinstance(model, KNNModel)
        model.create_model(**params)
        model.train(data["train_flat"], data["train_labels"])
        metrics = model.evaluate(data["test_flat"], data["test_labels"])
        artifact_path = None
    elif model_key == "cnn":
        assert isinstance(model, CNNModel)
        model.create_model(**params)
        default_config = CNNTrainingConfig()
        config = replace(
            default_config,
            epochs=200,
            patience=20,
            learning_rate=float(params.get("learning_rate", default_config.learning_rate)),
            weight_decay=float(params.get("weight_decay", default_config.weight_decay)),
            optimizer=params.get("optimizer", default_config.optimizer),
            batch_size=int(params.get("batch_size", default_config.batch_size)),
            checkpoint_path=str(out_dir / "best_model.pth"),
        )
        if device is None:
            device = get_device()
        model.train(
            data["train_images"],
            data["train_labels"],
            data["val_images"],
            data["val_labels"],
            config=config,
            device=device,
            verbose=True,
        )
        assert model.network is not None
        ckpt_path = out_dir / "best_model.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device if device else "cpu")
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.network.load_state_dict(state)
            # overwrite checkpoint with pure model state dict
            torch.save(model.network.state_dict(), ckpt_path)
        else:
            state = model.network.state_dict()
        metrics = model.evaluate(data["test_images"], data["test_labels"], device=device)
        artifact_path = out_dir / "model.pth"
        torch.save(model.network.state_dict(), artifact_path)
        if device is not None and device.type == "cuda":
            del model
            torch.cuda.empty_cache()
    metrics["composite_fitness"] = calculate_composite_fitness(metrics)
    return metrics, artifact_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Final training for best hyperparameters.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for final training runs.")
    parser.add_argument("--experiments", nargs="*", help="Optional list of experiment names (folders) to include.")
    parser.add_argument("--max-parallel-cnn", type=int, default=1, help="Max parallel CNN trainings.")
    parser.add_argument("--max-parallel-classic", type=int, default=1, help="Max parallel DT/KNN trainings.")
    args = parser.parse_args()

    set_seeds(args.seed)
    data = prepare_dataset()

    exp_dirs = [d for d in EXPERIMENT_ROOT.iterdir() if d.is_dir() and "-" in d.name]
    if args.experiments:
        exp_dirs = [d for d in exp_dirs if d.name in set(args.experiments)]

    def process_exp(exp_dir: Path):
        model_key: Literal["dt", "knn", "cnn"] = exp_dir.name.split("-")[0]  # type: ignore
        best_summary = find_best_run(exp_dir)
        best_params = best_summary.get("best_params", {})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = FINAL_ROOT / exp_dir.name / f"run_{timestamp}_{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Final training {exp_dir.name} ===")
        print(f"Using params: {best_params}")

        metrics, artifact = train_and_eval(model_key, best_params, data,  out_dir)

        summary = {
            "experiment": exp_dir.name,
            "seed": args.seed,
            "best_params": best_params,
            "test_metrics": metrics,
        }
        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to {out_dir / 'summary.json'}")
        if artifact:
            print(f"Saved model to {out_dir / artifact}")

    # Phase 1: CNN
    cnn_dirs = [d for d in exp_dirs if d.name.startswith("cnn-")]
    other_dirs = [d for d in exp_dirs if not d.name.startswith("cnn-")]

    if args.max_parallel_cnn <= 1 or not cnn_dirs:
        for d in sorted(cnn_dirs):
            process_exp(d)
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel_cnn) as executor:
            futures = {executor.submit(process_exp, d): d for d in sorted(cnn_dirs)}
            for future in as_completed(futures):
                _ = future.result()

    # Phase 2: DT/KNN
    if args.max_parallel_classic <= 1 or not other_dirs:
        for d in sorted(other_dirs):
            process_exp(d)
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel_classic) as executor:
            futures = {executor.submit(process_exp, d): d for d in sorted(other_dirs)}
            for future in as_completed(futures):
                _ = future.result()

    return 0


if __name__ == "__main__":
    main()


"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from framework import utils
from framework.datasets import CIFAR10Dataset


def load_cifar10_data():
    """Load CIFAR-10 dataset (grayscale from processed datasets)."""
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / ".cache" / "processed_datasets" / "cifar10"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. "
            "Please run:"
            "uv run python -m scripts.data_download"
            "uv run python -m scripts.data_process"
        )

    return load_from_disk(str(dataset_path))


def prepare_data(ds_dict, split: str):
    """Extract images and labels from dataset split."""
    ds = ds_dict[split]

    images = [np.asarray(img) for img in ds["image"]]
    labels = np.array(ds["label"])

    return images, labels


def split_train_val(
    images: List[np.ndarray],
    labels: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=val_ratio, stratify=labels, random_state=random_state
    )
    return X_train, y_train, X_val, y_val


def create_dataloaders(
    X_train: List[np.ndarray],
    y_train: np.ndarray,
    X_val: List[np.ndarray],
    y_val: np.ndarray,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CIFAR10Dataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=utils.is_cuda_available(),
    )

    val_dataset = CIFAR10Dataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=utils.is_cuda_available(),
    )

    return train_loader, val_loader

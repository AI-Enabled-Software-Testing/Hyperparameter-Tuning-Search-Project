"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from framework import utils
from framework.datasets import CIFAR10Dataset


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA image to grayscale.
    
    Args:
        image: Image array with shape (H, W, C) where C is 3 (RGB) or 4 (RGBA)
        
    Returns:
        Grayscale image with shape (H, W) 
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            # Single channel, just squeeze
            return image.squeeze(axis=2)
        elif image.shape[2] == 3:
            # RGB -> Grayscale using luminance weights
            # Y = 0.2125R + 0.7154G + 0.0721B
            return np.dot(image[...,:3], [0.2125, 0.7154, 0.0721])
        elif image.shape[2] == 4:
            # RGBA -> Grayscale (ignore alpha)
            return np.dot(image[...,:3], [0.2125, 0.7154, 0.0721])
    
    raise ValueError(f"Unsupported image shape: {image.shape}")


def preprocess_images_to_grayscale(images: List[np.ndarray]) -> List[np.ndarray]:
    """Convert a list of images to grayscale.
    
    Args:
        images: List of image arrays
        
    Returns:
        List of grayscale image arrays
    """
    return [convert_to_grayscale(img) for img in images]


def convert_dataset_to_grayscale(dataset):
    """Convert HuggingFace dataset images to grayscale in-place preprocessing.
    
    Args:
        dataset: HuggingFace dataset with 'image' column
        
    Returns:
        List of grayscale images and labels
    """
    images = []
    labels = []
    
    for item in dataset:
        img = np.array(item['image'])
        gray_img = convert_to_grayscale(img)
        images.append(gray_img)
        labels.append(item['label'])
    
    return images, np.array(labels)


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
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CIFAR10Dataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=utils.is_cuda_available(),
    )

    val_dataset = CIFAR10Dataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=utils.is_cuda_available(),
    )

    return train_loader, val_loader


def prepare_dataset(val_ratio: float = 0.1) -> Dict[str, Any]:
    """Prepare and return the CIFAR-10 dataset"""
    ds_dict = load_cifar10_data()
    train_images, train_labels = prepare_data(ds_dict, "train")
    test_images, test_labels = prepare_data(ds_dict, "test")

    X_train, y_train, X_val, y_val = split_train_val(
        train_images, train_labels, val_ratio=val_ratio
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

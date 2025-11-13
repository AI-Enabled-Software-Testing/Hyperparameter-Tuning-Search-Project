import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framework.data_utils import (
    create_dataloaders,
    load_cifar10_data,
    prepare_data,
    split_train_val,
)
from framework.utils import get_device, test_pytorch_setup
from models.cnn import CNNModel, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and save CIFAR-10 CNN")
    parser.add_argument(
        "--checkpoint-path",
        "--model-path",
        dest="checkpoint_path",
        type=str,
        default=".cache/models/cnn_cifar.pth",
        help="Where to store the best checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["AdamW", "Adam", "SGD"],
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional tensorboard log directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/cpu",
    )
    return parser.parse_args()


def train_model(args, writer: SummaryWriter):
    """Train the CNN model with the given arguments."""
    num_classes = 10

    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    print("Loading CIFAR-10 grayscale...")
    ds_dict = load_cifar10_data()

    print("Preparing training data...")
    X_all, y_all = prepare_data(ds_dict, "train")
    X_train, y_train, X_val, y_val = split_train_val(
        X_all, y_all, val_ratio=args.val_ratio
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_loader, val_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CNNModel(num_classes=num_classes)
    model.create_model()

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_path=Path(args.checkpoint_path),
        grad_clip_norm=args.grad_clip,
        batch_size=args.batch_size,
        writer=writer,
    )

    test_pytorch_setup()
    results = model.train(train_loader, val_loader, config=config, device=device)

    print("\nTraining complete!")
    print(
        f"Best val acc: {results['best_val_acc']:.4f} ({results['best_val_acc'] * 100:.2f}%)"
    )


def main():
    args = parse_args()

    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f".cache/tensorboard/cifar10_cnn_training/run_{timestamp}")
        log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    try:
        train_model(args, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

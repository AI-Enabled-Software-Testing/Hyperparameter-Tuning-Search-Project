import argparse
from pathlib import Path
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cnn import CNNModel, TrainingConfig
from framework.data_utils import load_cifar10_data, prepare_data
from framework import utils
from framework.utils import init_device, count_parameters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and save CIFAR-10 CNN")
    parser.add_argument(
        "--model-path",
        type=str,
        default=".cache/models/cnn_cifar.pth",
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
        default=0.0003
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
    weight_decay = 0.001
    val_ratio = 0.2
    num_classes = 10
    
    print("TensorBoard tracking initialized")
    
    init_device(args.device)
    print(f"Using device: {utils.device()}")

    print("Loading CIFAR-10 grayscale...")
    ds_dict = load_cifar10_data()

    print("Preparing training data...")
    X_all, y_all = prepare_data(ds_dict, "train")
    X_train, y_train, X_val, y_val = split_train_val(X_all, y_all, val_ratio=val_ratio)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )
    

    model = CNNModel(num_classes)
    model.create_model() 
    model = model.to(utils.device())

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    
    early_stopper = EarlyStopping(patience=15, min_delta=0.001)
    checkpoint = Checkpoint(args.model_path)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, utils.device(), scheduler=scheduler, aim_run=aim_run, epoch=epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, utils.device(), aim_run=aim_run, epoch=epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        if checkpoint.save_if_better(model, optimizer, epoch, val_acc, train_acc):
            print(f"Saved best model (val_acc={val_acc:.4f}) to {args.model_path}")

        if early_stopper(val_loss, val_acc):
            print(f"\nEarly stopping at {epoch}")
            print(f"Best val acc: {early_stopper.best_acc:.4f} ({early_stopper.best_acc*100:.2f}%)")
            break


    print("\nTraining complete!")
    print(f"Best val acc: {results['best_val_acc']:.4f} ({results['best_val_acc']*100:.2f}%)")


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f".cache/tensorboard/cifar10_cnn_training/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))
    
    try:
        train_model(args, writer)
    finally:
        writer.close()


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f".cache/tensorboard/cifar10_cnn_training/run_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))
    
    try:
        train_model(args, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cnn import CNNModel
from framework.data_utils import load_cifar10_data, prepare_data, split_train_val, create_dataloaders
from framework.training import EarlyStopping, train_epoch, validate, Checkpoint
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
        default=100,
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
    model = model.to(utils.device())

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log hyperparameters to TensorBoard
    hparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': 'AdamW',
        'weight_decay': weight_decay,
        'scheduler': 'OneCycleLR',
        'val_ratio': val_ratio,
        'num_classes': num_classes,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
    }
    
    # Log model graph (will be populated with metrics later)
    if writer is not None:
        try:
            # Get a sample input to trace the model
            sample_input = next(iter(train_loader))[0][:1].to(utils.device())
            writer.add_graph(model, sample_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

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
            model, train_loader, criterion, optimizer, utils.device(), 
            scheduler=scheduler, epoch=epoch, writer=writer
        )
        val_loss, val_acc = validate(model, val_loader, criterion, utils.device(), epoch=epoch, writer=writer)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        if checkpoint.save_if_better(model, optimizer, epoch, val_acc, train_acc):
            print(f"Saved best model (val_acc={val_acc:.4f}) to {args.model_path}")

        if early_stopper(val_loss, val_acc):
            print(f"\nEarly stopping at {epoch}")
            print(f"Best val acc: {early_stopper.best_acc:.4f} ({early_stopper.best_acc*100:.2f}%)")
            break
    
    # Log hyperparameters with final metrics
    if writer is not None:
        metrics = {
            'best_val_acc': checkpoint.best_val_acc,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
        }
        writer.add_hparams(hparams, metrics)

    print("\nTraining complete!")
    print(f"Best val acc: {checkpoint.best_val_acc:.4f} ({checkpoint.best_val_acc*100:.2f}%)")


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

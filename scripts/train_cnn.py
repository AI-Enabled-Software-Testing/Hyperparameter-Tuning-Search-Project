import argparse
from pathlib import Path
from datetime import datetime
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
    print(f"Total training samples: {len(X_all)}")

    model = CNNModel(num_classes)
    model.create_model() 
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
    
    cifar10_class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create training configuration
    config = TrainingConfig(
        writer=writer,
        weight_decay=weight_decay,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        optimizer='AdamW',
        val_ratio=val_ratio,
        checkpoint_path=args.model_path,
        early_stopping_min_delta=0.001,
        class_names=cifar10_class_names
    )
    
    # Train using the model's train() method with TensorBoard writer
    # The model will split X_all into train/val internally
    results = model.train(X_all, y_all, config)
    
    # Log hyperparameters with final metrics
    if writer is not None:
        metrics = {
            'best_val_acc': results['best_val_acc'],
            'final_train_acc': results['final_train_acc'],
            'final_val_acc': results['final_val_acc'],
        }
        writer.add_hparams(hparams, metrics)

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


if __name__ == "__main__":
    main()

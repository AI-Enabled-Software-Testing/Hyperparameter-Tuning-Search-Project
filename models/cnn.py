"""CNN model"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from framework.training import Checkpoint, EarlyStopping, train_epoch, validate
from framework.utils import count_parameters, get_device
from .ParamSpace import ParamSpace
from .base import BaseModel

MODEL_PATH = Path(".cache/models/cnn_cifar.pth")


@dataclass
class TrainingConfig:
    """Configuration for CNN model training."""

    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    optimizer: str = "AdamW"  # 'AdamW', 'Adam', 'SGD'
    patience: int = 15
    min_delta: float = 0.0
    checkpoint_path: Path = MODEL_PATH
    grad_clip_norm: float = 1.0
    writer: Optional[SummaryWriter] = None
    batch_size: int = 64


class Backbone(nn.Module):
    """CNN Model Backbone."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32, 64, kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                128,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class CNNModel(BaseModel):
    """Wrapper around the PyTorch CNN backbone."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.network: Optional[Backbone] = None
        self._input_channels = 1  # grayscale CIFAR-10

    def create_model(self, **params) -> None:
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        self.params.update(
            {
                "kernel_size": kernel_size,
                "stride": stride,
            }
        )
        self.network = Backbone(
            in_channels=self._input_channels,
            num_classes=self.num_classes,
            kernel_size=kernel_size,
            stride=stride,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        if self.network is None:
            raise RuntimeError("Train called before model is initialized")
        device = device or get_device()
        self.network = self.network.to(device)

        config = config or TrainingConfig()

        optimizer = self._build_optimizer(self.network, config)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
        )
        criterion = nn.CrossEntropyLoss()

        early_stopper = EarlyStopping(
            patience=config.patience, min_delta=config.min_delta
        )
        checkpoint = Checkpoint(str(config.checkpoint_path))

        total_params, trainable_params = count_parameters(self.network)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, config.epochs + 1):
            print(f"\nEpoch {epoch}/{config.epochs}")
            train_loss, train_acc = train_epoch(
                self.network,
                train_loader,
                criterion,
                optimizer,
                device,
                scheduler=scheduler,
                epoch=epoch,
                grad_clip_norm=config.grad_clip_norm,
                writer=config.writer,
            )
            val_loss, val_acc = validate(
                self.network,
                val_loader,
                criterion,
                device,
                epoch=epoch,
                writer=config.writer,
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)"
            )
            print(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc * 100:.2f}%)"
            )

            if checkpoint.save_if_better(
                self.network, optimizer, epoch, val_acc, train_acc
            ):
                print(
                    f"Saved best model (val_acc={val_acc:.4f}) to {config.checkpoint_path}"
                )

            if early_stopper(val_loss, val_acc):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print("\nTraining complete!")
        print(
            f"Best val acc: {checkpoint.best_val_acc:.4f} ({checkpoint.best_val_acc * 100:.2f}%)"
        )

        return {
            "best_val_acc": checkpoint.best_val_acc,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
        }

    def predict(
        self,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
        return_probabilities: bool = False,
    ) -> torch.Tensor:
        if self.network is None:
            raise RuntimeError("Predict called before model is initialized")
        device = device or get_device()
        network = self.network.to(device)
        network.eval()

        outputs = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                logits = network(images)
                if return_probabilities:
                    outputs.append(torch.softmax(logits, dim=1).cpu())
                else:
                    outputs.append(torch.argmax(logits, dim=1).cpu())
        return torch.cat(outputs, dim=0)

    def evaluate(
        self,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        if self.network is None:
            raise RuntimeError("Evaluate called before model is initialized")
        device = device or get_device()
        network = self.network.to(device)
        network.eval()

        criterion = criterion or nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = network(images)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples if total_samples else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            "kernel_size": ParamSpace.integer(min_val=3, max_val=5, default=3),
            "stride": ParamSpace.integer(min_val=1, max_val=3, default=1),
            "learning_rate": ParamSpace.float_range(
                min_val=1e-5, max_val=1e-2, default=3e-4
            ),
            "batch_size": ParamSpace.categorical(choices=[16, 32, 64, 128], default=64),
            "weight_decay": ParamSpace.float_range(
                min_val=0.0, max_val=0.01, default=1e-3
            ),
            "optimizer": ParamSpace.categorical(
                choices=["AdamW", "SGD"], default="AdamW"
            ),
        }

    def _build_optimizer(
        self, network: nn.Module, config: TrainingConfig
    ) -> optim.Optimizer:
        match config.optimizer:
            case "AdamW":
                return optim.AdamW(
                    network.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
            case "SGD":
                return optim.SGD(
                    network.parameters(),
                    lr=config.learning_rate,
                    momentum=0.9,
                    weight_decay=config.weight_decay,
                )
            case _:
                raise ValueError(f"Unsupported optimizer: {config.optimizer}")

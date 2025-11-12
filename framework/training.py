import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Optional
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter


class Checkpoint:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.best_val_acc = 0.0
    
    def save_if_better(
        self,
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        val_acc: float,
        train_acc: float,
        **kwargs
    ) -> bool:
        """Save checkpoint if validation accuracy improved."""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'train_acc': train_acc,
                **kwargs
            }
            torch.save(checkpoint_data, self.model_path)
            return True
        return False


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: Optional[LRScheduler] = None,
    epoch: int = 0,
    grad_clip_norm: float = 1.0,
    writer: Optional[SummaryWriter] = None,
    **kwargs
) -> Tuple[float, float]:
    """Trains the model for one epoch and returns the epoch loss and accuracy."""
    nn.Module.train(model, mode=True)
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = loss.item()
        running_loss += loss_value
        
        _, predicted = torch.max(outputs.data, 1)
        batch_correct = (predicted == labels).sum().item()
        total += labels.size(0)
        correct += batch_correct

        if batch_idx % 10 == 0 and writer is not None:
            batch_total = labels.size(0)
            batch_acc = 100 * batch_correct / batch_total
            current_lr = optimizer.param_groups[0]['lr']
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss_value, step)
            writer.add_scalar('train/batch_accuracy', batch_acc, step)
            writer.add_scalar('train/learning_rate', current_lr, step)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    if writer is not None:
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        writer.add_scalar('train/epoch_accuracy', epoch_acc * 100, epoch)

    return epoch_loss, epoch_acc


def validate(
    model: Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int = 0,
    writer: Optional[SummaryWriter] = None,
    **kwargs
) -> Tuple[float, float]:
    """Validates the model and returns the epoch loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total

    if writer is not None:
        writer.add_scalar('val/epoch_loss', epoch_loss, epoch)
        writer.add_scalar('val/epoch_accuracy', epoch_acc * 100, epoch)

    return epoch_loss, epoch_acc
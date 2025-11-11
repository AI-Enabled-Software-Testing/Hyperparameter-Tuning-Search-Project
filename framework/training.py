import torch
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
try:
    from aim import Run
    from aim.pytorch import track_params_dists, track_gradients_dists
except ImportError:
    print("Aim is not installed. Metrics tracking will be disabled.")
    pass


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
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                **kwargs
            }
            
            torch.save(checkpoint, str(self.model_path))
            return True
        return False
    
    def load(self, model: Module, optimizer: Optimizer) -> dict:
        """Load checkpoint from disk."""
        checkpoint = torch.load(str(self.model_path), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        return checkpoint


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_acc = 0.0
    
    def __call__(self, val_loss: float, val_acc: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.counter = 0
            return False
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
    aim_run: Run,
    scheduler: LRScheduler,
    epoch: int = 0,
    grad_clip_norm: float = 1.0
) -> Tuple[float, float]:
    """Trains the model for one epoch and returns the epoch loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()

        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Track metrics
        if batch_idx % 10 == 0:
            # Batch accuracy
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            batch_acc = 100 * batch_correct / batch_total
            current_lr = optimizer.param_groups[0]['lr']
            aim_run.track(
                {
                    'loss': loss.item(),
                    'accuracy': batch_acc,
                    'learning_rate': current_lr
                },
                step=epoch * len(train_loader) + batch_idx,
                context={'subset': 'train'}
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    # Track epoch-level metrics
    if aim_run is not None and 'aim' in globals():
        aim_run.track(
            {
                'loss': epoch_loss,
                'accuracy': epoch_acc * 100
            },
            step=epoch,
            context={'subset': 'train'}
        )

        track_params_dists(model, aim_run)
        track_gradients_dists(model, aim_run)

    return epoch_loss, epoch_acc


def validate(
    model: Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    aim_run: Run,
    epoch: int = 0
) -> Tuple[float, float]:
    """Validates the model and returns the epoch loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) # forward pass
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total

    if aim_run is not None and 'aim' in globals():
        aim_run.track(
            {
                'loss': epoch_loss,
                'accuracy': epoch_acc * 100
            },
            step=epoch,
            context={'subset': 'val'}
        )

    return epoch_loss, epoch_acc


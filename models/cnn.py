import torch
import torch.nn as nn
from typing import Dict

from framework.utils import count_parameters
from .ParamSpace import ParamSpace
from .base import BaseModel
from framework.data_utils import create_dataloaders
from framework.training import EarlyStopping, Checkpoint, train_epoch, validate
from framework import utils

import torch.optim as optim

from sklearn.model_selection import train_test_split

MODEL_PATH = ".cache/models/cnn_cifar.pth"

class CNNModel(nn.Module, BaseModel):   
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def create_model(self, **params):
        # Extract hyperparameters with defaults
        kernel_size = params.get('kernel_size', 4)
        stride = params.get('stride', 2)
        padding = params.get('padding', 1)
        
        features = nn.Sequential(
            # Conv1: 32×32×1 -> 16×16×32
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv2: 16×16×32 -> 8×8×64
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv3: 8×8×64 -> 4×4×128
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 4×4×128 -> 1×1×128
        )
        
        self.model = nn.Sequential(
            nn.Flatten(),  # 1×1×128 -> 128
            nn.Dropout(),
            nn.Linear(128, self.num_classes)
        )

        # Optional: Compile the model for computational graph optimization
        self.model = torch.compile(self.model)
        self.features = torch.compile(features)
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call create_model() first.")
        
        x = self.features(x)
        x = self.model(x)
        return x
    
    def train(self, X_train, y_train, weight_decay=0.001, epochs=100, learning_rate=0.001, batch_size=32, optimizer='AdamW', scheduler='OneCycleLR', val_ratio=0.2):
        """Train the CNN model with PyTorch training loop."""
        self = self.to(utils.device()) # Move model to device
        # Check if model is instantiated properly
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call create_model() first.")

        # Log Parameters
        total_params, trainable_params = count_parameters(self)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Break training set into train + val according to ratio
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio)

        # Convert to tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.tensor(y_val, dtype=torch.long)
        
        # Create data loader
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=batch_size
        )

        # Setup Scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        # Setup early stopping
        early_stopper = EarlyStopping(patience=15, min_delta=0.001)
        checkpoint = Checkpoint(MODEL_PATH)

        # Setup optimizer and loss
        match optimizer:
            case 'AdamW':
                optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case 'Adam':
                optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        self.train_mode = True  # Set to training mode
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            total_loss = 0
            train_loss, train_acc = train_epoch(
                self, train_loader, criterion, optimizer, utils.device(), scheduler=scheduler, aim_run=None, epoch=epoch
            )
            val_loss, val_acc = validate(self, val_loader, criterion, utils.device(), aim_run=None, epoch=epoch)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            if checkpoint.save_if_better(self, optimizer, epoch, val_acc, train_acc):
                print(f"Saved best model (val_acc={val_acc:.4f}) to {MODEL_PATH}")

            if early_stopper(val_loss, val_acc):
                print(f"\nEarly stopping at {epoch}")
                print(f"Best val acc: {early_stopper.best_acc:.4f} ({early_stopper.best_acc*100:.2f}%)")
                break

            if epoch % (epochs // 10) == 0:  # Print every 10% of training
                print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss/(len(train_loader) + len(val_loader)):.4f}')
        

        print("\nTraining complete!")
        print(f"Best val acc: {checkpoint.best_val_acc:.4f} ({checkpoint.best_val_acc*100:.2f}%)")
        return self

    def predict(self, X_test, return_proba=False):
        """Make predictions with the CNN model."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call create_model() first.")
        
        # Convert to tensor if needed
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        
        self.eval()  # Set to evaluation mode
        with torch.no_grad(): # without modifying gradients
            outputs = self.forward(X_test)
            
            if return_proba:
                # Return probabilities
                probabilities = torch.softmax(outputs, dim=1)
                return probabilities.numpy()
            else:
                # Return class predictions
                _, predicted = torch.max(outputs, 1)
                return predicted.numpy()
    
    def set_train_mode(self, mode: bool = True):
        """Override PyTorch's train method to handle compiled models."""
        super().train(mode)
        if self.is_initialized:
            self.features.train(mode)
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'kernel_size': ParamSpace.integer(min_val=3, max_val=7, default=4),
            'stride': ParamSpace.integer(min_val=1, max_val=3, default=2),
            'padding': ParamSpace.integer(min_val=0, max_val=2, default=1),
            'epochs': ParamSpace.integer(min_val=10, max_val=200, default=100),
            'learning_rate': ParamSpace.float_range(min_val=1e-5, max_val=1e-1, default=0.001, log_scale=True),
            'batch_size': ParamSpace.integer(min_val=16, max_val=128, default=32),
        }

    @property
    def is_initialized(self) -> bool:
        return hasattr(self, 'features') and hasattr(self, 'model')



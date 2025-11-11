import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from framework.utils import count_parameters
from .ParamSpace import ParamSpace
from .base import BaseModel
from framework.data_utils import create_dataloaders
from framework.training import EarlyStopping, Checkpoint, train_epoch, validate
from framework import utils

import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

MODEL_PATH = ".cache/models/cnn_cifar.pth"

@dataclass
class TrainingConfig:
    """Configuration for CNN model training."""
    writer: SummaryWriter
    weight_decay: float = 0.001
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = 'AdamW'  # 'AdamW', 'Adam', 'SGD'
    val_ratio: float = 0.2
    checkpoint_path: Optional[str] = None
    early_stopping_min_delta: float = 0.001
    class_names: Optional[list] = None
    sgd_momentum: float = 0.9  # Momentum for SGD optimizer

# PyTorch Setup
print(f"Using Device: {utils.device()}")
print(f"Is CUDA Available: {utils.is_cuda()}")

class CNNModel(nn.Module, BaseModel):   
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        # Initialize metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def create_model(self, **params):
        # Extract hyperparameters with defaults
        kernel_size = params.get('kernel_size', 4)
        stride = params.get('stride', 2)
        padding = params.get('padding', 1)
        
        self.features = nn.Sequential(
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

        # Print model architecture
        print("CNN Model Architecture:")
        print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call create_model() first.")
        
        x = self.features(x)
        x = self.model(x)
        return x
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training and validation loss and accuracy curves.
        
        Args:
            save_path (str): Path to save the plot. If None, saves to results/ folder with timestamp.
        """        
        if not self.train_losses:
            print("No training history found. Train the model first.")
            return
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot loss
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, [acc * 100 for acc in self.train_accuracies], 'b-', 
                label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, [acc * 100 for acc in self.val_accuracies], 'r-', 
                label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"results/cnn_training_history_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        
        # Also save training history as text file
        history_path = save_path.replace('.png', '_data.txt')
        with open(history_path, 'w') as f:
            f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")
            for i, (tl, ta, vl, va) in enumerate(zip(self.train_losses, self.train_accuracies, 
                                                   self.val_losses, self.val_accuracies), 1):
                f.write(f"{i},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f}\n")
        print(f"Training history data saved to: {history_path}")
    
    def reset_metrics(self):
        """Reset all training metrics."""
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def plot_confusion_matrices(self, X_train, y_train, X_val, y_val, class_names=None, save_path=None):
        """
        Generate and save confusion matrices for training and validation sets.
        
        Args:
            X_train: Training features tensor
            y_train: Training labels tensor  
            X_val: Validation features tensor
            y_val: Validation labels tensor
            class_names: List of class names for labeling (optional)
            save_path: Path to save the plot. If None, saves to results/ folder with timestamp.
        """
        if not self.is_initialized:
            print("Model not initialized. Cannot generate confusion matrices.")
            return
            
        # Get predictions for both sets
        self.eval()
        with torch.no_grad():
            # Training predictions - fix shape from [1, N, 32, 32] to [N, 1, 32, 32]
            X_train = X_train.squeeze(0).unsqueeze(1).to(utils.device())
            train_outputs = self(X_train)
            _, train_pred = torch.max(train_outputs, 1)
            train_pred = train_pred.cpu().numpy()
            y_train_np = y_train.cpu().numpy() if hasattr(y_train, 'cpu') else y_train
            
            # Validation predictions - fix shape from [1, N, 32, 32] to [N, 1, 32, 32]
            X_val = X_val.squeeze(0).unsqueeze(1).to(utils.device())
            val_outputs = self(X_val)
            _, val_pred = torch.max(val_outputs, 1)
            val_pred = val_pred.cpu().numpy()
            y_val_np = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else y_val
        
        # Compute confusion matrices
        train_cm = confusion_matrix(y_train_np, train_pred)
        val_cm = confusion_matrix(y_val_np, val_pred)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Set class names
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # Plot training confusion matrix
        self._plot_confusion_matrix(train_cm, ax1, class_names, 'Training Set Confusion Matrix')
        
        # Plot validation confusion matrix  
        self._plot_confusion_matrix(val_cm, ax2, class_names, 'Validation Set Confusion Matrix')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"results/cnn_confusion_matrices_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {save_path}")
        
        # Save confusion matrix data
        data_path = save_path.replace('.png', '_data.txt')
        with open(data_path, 'w') as f:
            f.write("Training Confusion Matrix:\n")
            f.write("Predicted\\Actual," + ",".join(class_names) + "\n")
            for i, row in enumerate(train_cm):
                f.write(f"{class_names[i]}," + ",".join(map(str, row)) + "\n")
            
            f.write("\nValidation Confusion Matrix:\n")
            f.write("Predicted\\Actual," + ",".join(class_names) + "\n")
            for i, row in enumerate(val_cm):
                f.write(f"{class_names[i]}," + ",".join(map(str, row)) + "\n")
        
        print(f"Confusion matrix data saved to: {data_path}")
    
    def _plot_confusion_matrix(self, cm, ax, class_names, title):
        """Helper method to plot a single confusion matrix."""
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Set ticks and labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=9)
        
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
    
    def _confusion_matrix_to_image(self, cm, class_names, title):
        """Convert a confusion matrix to a numpy array image for TensorBoard."""
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_confusion_matrix(cm, ax, class_names, title)
        plt.tight_layout()
        
        # Convert figure to numpy array using canvas
        fig.canvas.draw()
        # Get the buffer as RGBA array
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        img_array = buf.reshape((height, width, 4))[:, :, :3]  # Remove alpha channel, keep RGB
        
        # Convert RGB to CHW format for TensorBoard (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        plt.close(fig)
        
        return img_array
    
    def evaluate_and_plot_confusion_matrix(self, X_test, y_test, dataset_name="Test", class_names:list=None, save_path=None):
        """
        Evaluate the model on a dataset and plot confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of the dataset (for plot title)
            class_names: List of class names for labeling (optional)
            save_path: Path to save the plot. If None, saves to results/ folder with timestamp.
        """
        if not self.is_initialized:
            print("Model not initialized. Cannot evaluate.")
            return
        
        # Get predictions
        predictions = self.predict(X_test, return_proba=False)
        
        # Convert labels to numpy if needed
        if hasattr(y_test, 'cpu'):
            y_test_np = y_test.cpu().numpy()
        elif hasattr(y_test, 'numpy'):
            y_test_np = y_test.numpy()
        else:
            y_test_np = y_test
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test_np, predictions)
        
        # Calculate accuracy
        accuracy = np.sum(predictions == y_test_np) / len(y_test_np)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Set class names
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # Plot confusion matrix
        title = f'{dataset_name} Set Confusion Matrix (Accuracy: {accuracy:.3f})'
        self._plot_confusion_matrix(cm, ax, class_names, title)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"results/cnn_{dataset_name.lower()}_confusion_matrix_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{dataset_name} confusion matrix saved to: {save_path}")
        
        # Save confusion matrix data
        data_path = save_path.replace('.png', '_data.txt')
        with open(data_path, 'w') as f:
            f.write(f"{dataset_name} Confusion Matrix (Accuracy: {accuracy:.6f}):\n")
            f.write("Predicted\\Actual," + ",".join(class_names) + "\n")
            for i, row in enumerate(cm):
                f.write(f"{class_names[i]}," + ",".join(map(str, row)) + "\n")
        
        print(f"{dataset_name} confusion matrix data saved to: {data_path}")
                
        return accuracy, cm
    
    def train(self, X_train=None, y_train=None, config: TrainingConfig=None, mode=True):
        """
            Train the CNN model with PyTorch training loop, or set training mode.
            
            If called with only mode parameter (boolean), sets the model to training/eval mode (PyTorch behavior).
            If called with X_train, y_train, config, performs full training.
            
            Params:
                X_train: Training features (must be a numpy array or tensor)
                y_train: Training labels (must be a numpy array or tensor)
                config: TrainingConfig dataclass containing all training parameters
                mode: Boolean to set training mode (for PyTorch compatibility)
                
            Returns:
                dict: Training results with checkpoint info and final metrics (if training)
                self: The model itself (if setting mode)
        """
        # Handle PyTorch's train() call for training mode
        # When eval() calls train(False), X_train will be False (bool) and y_train/config will be None
        # When train() is called normally, X_train will be None and mode defaults to True
        if (isinstance(X_train, bool) or (X_train is None and y_train is None and config is None)):
            mode_value = X_train if isinstance(X_train, bool) else mode
            return super().train(mode_value)
        
        # Reset metrics for new training session
        self.reset_metrics()
        
        self = self.to(utils.device()) # Move model to device
        # Check if model is instantiated properly
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call create_model() first.")

        # Log Parameters
        self.total_params, self.trainable_params = count_parameters(self)
        print(f"Total parameters: {self.total_params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")

        # Break training set into train + val according to ratio
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config.val_ratio)

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
            X_train, y_train, X_val, y_val, batch_size=config.batch_size, num_workers=2
        )

        # Setup optimizer and loss first (before scheduler)
        match config.optimizer:
            case 'AdamW': # AdamW optimizer with weight decay
                optimizer_obj = optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            case 'Adam':
                optimizer_obj = optim.Adam(self.parameters(), lr=config.learning_rate)
            case 'SGD': # Stochastic Gradient Descent with momentum
                optimizer_obj = optim.SGD(self.parameters(), lr=config.learning_rate, momentum=config.sgd_momentum)
            case _:
                raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        criterion = torch.nn.CrossEntropyLoss()

        # Setup Scheduler (after optimizer is created)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer_obj,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
        )

        early_stopper = EarlyStopping(patience=100, min_delta=config.early_stopping_min_delta)
        checkpoint_path = config.checkpoint_path or MODEL_PATH
        checkpoint = Checkpoint(checkpoint_path)
        
        for epoch in range(1, config.epochs + 1):
            print(f"\nEpoch {epoch}/{config.epochs}")
            total_loss = 0
            train_loss, train_acc = train_epoch(
                self, train_loader, criterion, optimizer_obj, utils.device(), 
                scheduler, epoch, 1.0, config.writer
            )
            val_loss, val_acc = validate(self, val_loader, criterion, utils.device(), epoch, config.writer)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            if checkpoint.save_if_better(self, optimizer_obj, epoch, val_acc, train_acc):
                print(f"Saved best model (val_acc={val_acc:.4f}) to {checkpoint_path}")

            # Log confusion matrices to TensorBoard every 10 epochs
            if epoch % 10 == 0 and config.writer is not None:
                self.eval()
                with torch.no_grad():
                    # Get predictions for training set
                    X_train_tensor = X_train.squeeze(0).unsqueeze(1).to(utils.device())
                    train_outputs = self(X_train_tensor)
                    _, train_pred = torch.max(train_outputs, 1)
                    train_pred = train_pred.cpu().numpy()
                    y_train_np = y_train.cpu().numpy() if hasattr(y_train, 'cpu') else y_train
                    
                    # Get predictions for validation set
                    X_val_tensor = X_val.squeeze(0).unsqueeze(1).to(utils.device())
                    val_outputs = self(X_val_tensor)
                    _, val_pred = torch.max(val_outputs, 1)
                    val_pred = val_pred.cpu().numpy()
                    y_val_np = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else y_val
                
                # Compute confusion matrices
                train_cm = confusion_matrix(y_train_np, train_pred)
                val_cm = confusion_matrix(y_val_np, val_pred)
                
                # Convert to images and log to TensorBoard
                class_names = config.class_names if config.class_names is not None else [f'Class {i}' for i in range(self.num_classes)]
                train_cm_img = self._confusion_matrix_to_image(train_cm, class_names, f'Training Confusion Matrix - Epoch {epoch}')
                val_cm_img = self._confusion_matrix_to_image(val_cm, class_names, f'Validation Confusion Matrix - Epoch {epoch}')
                
                config.writer.add_image('confusion_matrix/train', train_cm_img, epoch)
                config.writer.add_image('confusion_matrix/val', val_cm_img, epoch)

            if early_stopper(val_loss, val_acc):
                print(f"\nEarly stopping at {epoch}")
                print(f"Best val acc: {early_stopper.best_acc:.4f} ({early_stopper.best_acc*100:.2f}%)")
                break

            if epoch % (config.epochs // 10) == 0:  # Print every 10% of training
                print(f'Epoch [{epoch}/{config.epochs}], Loss: {total_loss/(len(train_loader) + len(val_loader)):.4f}')
        

        print("\nTraining complete!")
        print(f"Best val acc: {checkpoint.best_val_acc:.4f} ({checkpoint.best_val_acc*100:.2f}%)")
        
        # Create and save training history plot
        save_path = os.path.join(
            ".cache",
            "results"
        )
        os.makedirs(save_path, exist_ok=True)
        self.plot_training_history(save_path=os.path.join(
            save_path,
            f"cnn_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ))
        
        # Create and save confusion matrices
        self.plot_confusion_matrices(X_train, y_train, X_val, y_val, class_names=config.class_names, save_path=os.path.join(
            save_path,
            f"cnn_confusion_matrices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ))

        return {
            'model': self,
            'checkpoint': checkpoint,
            'best_val_acc': checkpoint.best_val_acc,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
        }

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
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'kernel_size': ParamSpace.integer(min_val=3, max_val=7, default=4),
            'stride': ParamSpace.integer(min_val=1, max_val=3, default=2),
            'padding': ParamSpace.integer(min_val=0, max_val=2, default=1),
            'epochs': ParamSpace.integer(min_val=10, max_val=200, default=100),
            'learning_rate': ParamSpace.float_range(min_val=1e-5, max_val=1e-1, default=0.001),
            'batch_size': ParamSpace.integer(min_val=16, max_val=128, default=32),
            'weight_decay': ParamSpace.float_range(min_val=0.0, max_val=0.01, default=0.001),
            'optimizer': ParamSpace.categorical(choices=['AdamW', 'Adam', 'SGD'], default='AdamW'), # For SGD, momentum is fixed at 0.9
        }

    @property
    def is_initialized(self) -> bool:
        return hasattr(self, 'features') and hasattr(self, 'model')



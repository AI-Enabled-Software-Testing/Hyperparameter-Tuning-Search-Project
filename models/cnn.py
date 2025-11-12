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
from torch.utils.data import TensorDataset, DataLoader

from framework.utils import count_parameters
from .ParamSpace import ParamSpace
from .base import BaseModel
from framework.training import EarlyStopping, Checkpoint, train_epoch, validate
from framework import utils

import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import math

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
    def __init__(self, num_classes: int = 10, input_channels: int = None):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels  # Will be determined automatically from data
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
        
        # Use detected input channels or default to 1 for grayscale
        in_channels = self.input_channels if self.input_channels is not None else 1
        
        # Validate kernel size compatibility (prevent kernel larger than input after conv layers)
        self._validate_architecture_params(kernel_size, stride, padding)
        
        # Use adaptive architecture for different input sizes
        if hasattr(self, '_input_height') and hasattr(self, '_input_width'):
            # For small inputs (MNIST: 28x28), use smaller kernels and strides
            if self._input_height <= 28 or self._input_width <= 28:
                kernel_size = min(kernel_size, 3)  # Cap at 3x3 for small inputs
                stride = min(stride, 2)  # Cap stride at 2 for small inputs
        
        self.features = nn.Sequential(
            # Conv1: Handle variable input channels
            nn.Conv2d(in_channels, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv2: 32 -> 64 (use smaller kernel for subsequent layers)
            nn.Conv2d(32, 64, kernel_size=min(kernel_size, 3), stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv3: 64 -> 128 (use even smaller kernel for final conv)
            nn.Conv2d(64, 128, kernel_size=min(kernel_size, 3), stride=max(1, stride-1), padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Adaptive pooling to 1×1×128
        )
        
        self.model = nn.Sequential(
            nn.Flatten(),  # 1×1×128 -> 128
            nn.Dropout(),
            nn.Linear(128, self.num_classes)
        )

        # Print model architecture
        print(f"CNN Model Architecture (input channels: {in_channels}, kernel: {kernel_size}, stride: {stride}):")
        print(self)

    def _validate_architecture_params(self, kernel_size, stride, padding):
        """Validate that the CNN architecture parameters are compatible with expected input sizes."""
        def calc_output_size(input_size, kernel, stride, padding):
            return (input_size + 2 * padding - kernel) // stride + 1
        
        # Test with common input sizes (MNIST: 28x28, CIFAR-10: 32x32)
        test_sizes = [28, 32]  # Height/Width dimensions
        
        for input_size in test_sizes:
            # Simulate 3 conv layers
            size = input_size
            for layer in range(3):
                # Use progressive kernel size reduction as in create_model
                if layer == 0:
                    k = kernel_size
                elif layer == 1:
                    k = min(kernel_size, 3)
                else:  # layer == 2
                    k = min(kernel_size, 3)
                    s = max(1, stride-1)
                    size = calc_output_size(size, k, s, padding)
                    continue
                
                size = calc_output_size(size, k, stride, padding)
                
                if size <= 0:
                    raise ValueError(
                        f"Invalid CNN parameters: kernel_size={kernel_size}, stride={stride}, padding={padding}. "
                        f"Results in non-positive output size ({size}) at layer {layer+1} for {input_size}x{input_size} input."
                    )
                elif size < k and layer < 2:  # Check if next layer's kernel will fit
                    next_k = min(kernel_size, 3) if layer == 0 else min(kernel_size, 3)
                    if size < next_k:
                        raise ValueError(
                            f"Invalid CNN parameters: kernel_size={kernel_size}, stride={stride}, padding={padding}. "
                            f"Layer {layer+1} output size ({size}) too small for next layer kernel ({next_k}) "
                            f"for {input_size}x{input_size} input."
                        )

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
            # Training predictions
            X_train_device = X_train.to(utils.device())
            train_outputs = self(X_train_device)
            _, train_pred = torch.max(train_outputs, 1)
            train_pred = train_pred.cpu().numpy()
            y_train_np = y_train.cpu().numpy() if hasattr(y_train, 'cpu') else y_train
            
            # Validation predictions
            X_val_device = X_val.to(utils.device())
            val_outputs = self(X_val_device)
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
    
    def _preprocess_input_data(self, X, dataset_name="Unknown"):
        """
        Preprocess input data to handle different input formats from the notebook.
        Converts flattened arrays back to proper image tensors with correct channels.
        """
        print(f"Preprocessing input data for {dataset_name}...")
        
        # Convert to numpy array if it's a tensor
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.array(X)
        
        print(f"Original shape: {X_np.shape}")
        
        # Handle different input shapes
        if len(X_np.shape) == 2:  # Flattened data: (n_samples, flattened_pixels)
            n_samples = X_np.shape[0]
            total_pixels = X_np.shape[1]
            
            # Infer image dimensions and channels
            if dataset_name.upper() == "MNIST" or total_pixels == 784:  # MNIST: 28x28x1 = 784
                height, width, channels = 28, 28, 1
                X_np = X_np.reshape(n_samples, height, width, channels)
                print(f"Detected MNIST format: reshaped to {X_np.shape}")
            elif dataset_name.upper() == "CIFAR-10" or total_pixels == 3072:  # CIFAR-10: 32x32x3 = 3072
                height, width, channels = 32, 32, 3
                X_np = X_np.reshape(n_samples, height, width, channels)
                print(f"Detected CIFAR-10 format: reshaped to {X_np.shape}")
            elif total_pixels == 1024:  # 32x32x1 = 1024 (grayscale version of CIFAR-10)
                height, width, channels = 32, 32, 1
                X_np = X_np.reshape(n_samples, height, width, channels)
                print(f"Detected 32x32 grayscale format: reshaped to {X_np.shape}")
            else:
                # Try to infer square image dimensions
                # Assume grayscale first
                side = int(math.sqrt(total_pixels))
                if side * side == total_pixels:
                    height, width, channels = side, side, 1
                    X_np = X_np.reshape(n_samples, height, width, channels)
                    print(f"Inferred square grayscale: reshaped to {X_np.shape}")
                else:
                    # Try RGB
                    pixels_per_channel = total_pixels // 3
                    side = int(math.sqrt(pixels_per_channel))
                    if side * side * 3 == total_pixels:
                        height, width, channels = side, side, 3
                        X_np = X_np.reshape(n_samples, height, width, channels)
                        print(f"Inferred square RGB: reshaped to {X_np.shape}")
                    else:
                        raise ValueError(f"Cannot infer image dimensions from flattened shape {X_np.shape}")
        
        elif len(X_np.shape) == 3:  # (n_samples, height, width) - grayscale
            n_samples, height, width = X_np.shape
            channels = 1
            X_np = X_np.reshape(n_samples, height, width, channels)
            print(f"Added channel dimension: reshaped to {X_np.shape}")
        
        elif len(X_np.shape) == 4:  # (n_samples, height, width, channels) - already proper format
            n_samples, height, width, channels = X_np.shape
            print(f"Already in correct format: {X_np.shape}")
        
        else:
            raise ValueError(f"Unsupported input shape: {X_np.shape}")
        
        # Convert to PyTorch tensor format: (N, C, H, W)
        X_tensor = torch.from_numpy(X_np).permute(0, 3, 1, 2).float()
        
        # Store input dimensions for architecture validation
        _, channels, height, width = X_tensor.shape
        self._input_height = height
        self._input_width = width
        
        # Normalize pixel values to [0, 1] if they're in [0, 255] range
        if X_tensor.max() > 1.0:
            X_tensor = X_tensor / 255.0
            print("Normalized pixel values to [0, 1] range")
        
        # Update the model's input channels if not set
        if self.input_channels is None:
            self.input_channels = channels
            print(f"Detected {channels} input channel(s), input size: {height}x{width}")
            # Only recreate model if it hasn't been trained yet (no trained weights)
            if hasattr(self, 'features') and not self._has_trained_weights():
                print("Recreating model with correct input channels...")
                self.create_model()
            elif hasattr(self, 'features') and self._has_trained_weights():
                print(f"Warning: Model was already trained, keeping existing weights despite channel mismatch.")
                print(f"Expected {channels} channels, model has {self.features[0].in_channels} channels.")
                # Don't recreate to preserve trained weights
        
        print(f"Final tensor shape: {X_tensor.shape} (N, C, H, W)")
        return X_tensor
    
    def train(self, X_train=None, y_train=None, class_names=None, config: TrainingConfig=None, mode=True):
        """
            Train the CNN model with PyTorch training loop, or set training mode.
            
            If called with only mode parameter (boolean), sets the model to training/eval mode (PyTorch behavior).
            If called with X_train, y_train, class_names, config, performs full training.
            
            Params:
                X_train: Training features (must be a numpy array or tensor)
                y_train: Training labels (must be a numpy array or tensor)
                class_names: List of class names for the dataset (optional)
                config: TrainingConfig dataclass containing all training parameters
                mode: Boolean to set training mode (for PyTorch compatibility)
                
            Returns:
                dict: Training results with checkpoint info and final metrics (if training)
                self: The model itself (if setting mode)
        """
        # Handle PyTorch's train() call for training mode
        # When eval() calls train(False), X_train will be False (bool) and y_train/config will be None
        # When train() is called normally, X_train will be None and mode defaults to True
        if (isinstance(X_train, bool) or (X_train is None and y_train is None and class_names is None and config is None)):
            mode_value = X_train if isinstance(X_train, bool) else mode
            return super().train(mode_value)
        
        # Reset metrics for new training session
        self.reset_metrics()
        
        self = self.to(utils.device()) # Move model to device
        # Check if model is instantiated properly
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call create_model() first.")

        # Preprocess input data to handle different formats from notebook
        X_train = self._preprocess_input_data(X_train, "training data")

        # Handle case where config is None - create default config
        if config is None:
            from torch.utils.tensorboard import SummaryWriter
            config = TrainingConfig(
                writer=SummaryWriter(),
                val_ratio=0.2,
                epochs=10,  # Reduced default for quick testing
                learning_rate=0.001,
                batch_size=32,
                optimizer='AdamW',
                weight_decay=0.001,
                early_stopping_min_delta=0.001,
                class_names=class_names  # Use the passed class_names
            )
            print("No config provided, using default training configuration.")

        # Log Parameters
        self.total_params, self.trainable_params = count_parameters(self)
        print(f"Total parameters: {self.total_params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")

        # Break training set into train + val according to ratio
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config.val_ratio)

        # Convert labels to tensors if needed
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.tensor(y_val, dtype=torch.long)
        
        # Create PyTorch datasets and dataloaders
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=utils.is_cuda()
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=utils.is_cuda()
        )
        
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
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
            train_loss, train_acc = train_epoch(
                self, train_loader, criterion, optimizer_obj, utils.device(), 
                scheduler, epoch, 1.0, config.writer
            )
            val_loss, val_acc = validate(self, val_loader, criterion, utils.device(), epoch, config.writer)
            total_loss = train_loss + val_loss
            
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
                    X_train_device = X_train.to(utils.device())
                    train_outputs = self(X_train_device)
                    _, train_pred = torch.max(train_outputs, 1)
                    train_pred = train_pred.cpu().numpy()
                    y_train_np = y_train.cpu().numpy() if hasattr(y_train, 'cpu') else y_train
                    
                    # Get predictions for validation set
                    X_val_device = X_val.to(utils.device())
                    val_outputs = self(X_val_device)
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
        
        # Preprocess input data to handle different formats from notebook
        X_test = self._preprocess_input_data(X_test, "test data")
        
        # Move to device
        X_test = X_test.to(utils.device())
        
        self.eval()  # Set to evaluation mode
        with torch.no_grad(): # without modifying gradients
            outputs = self.forward(X_test)
            
            if return_proba:
                # Return probabilities
                probabilities = torch.softmax(outputs, dim=1)
                return probabilities.cpu().numpy()
            else:
                # Return class predictions
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
    
    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate the CNN model using custom predict method."""
        # Define default metrics dictionary - 0.0 for most metrics, 0.5 for ROC AUC (random chance)
        metric_names = ["accuracy", "precision", "recall", "F1 (Macro)", "F1 (Micro)", "ROC AUC"]
        metrics = {name: (0.5 if name == "ROC AUC" else 0.0) for name in metric_names}
        
        if not self.is_initialized:
            return metrics
        
        # Get predictions using CNN's custom predict method
        y_pred = self.predict(X_test, return_proba=False)
        
        # Ensure we have valid predictions
        if len(y_pred) == 0 or len(y_test) == 0:
            return metrics
        
        # Convert y_test to numpy if needed
        if hasattr(y_test, 'cpu'):
            y_test = y_test.cpu().numpy()
        elif hasattr(y_test, 'numpy'):
            y_test = y_test.numpy()
        
        # Calculate basic accuracy
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import warnings
        
        accuracy = accuracy_score(y_test, y_pred)
        metrics["accuracy"] = accuracy
        
        # Calculate advanced metrics safely
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                unique_true = np.unique(y_test)
                unique_pred = np.unique(y_pred)
                
                if len(unique_true) > 1 and len(unique_pred) > 1:
                    metrics["precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics["recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics["F1 (Macro)"] = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    metrics["F1 (Micro)"] = f1_score(y_test, y_pred, average='micro', zero_division=0)
                    
                    # ROC AUC for multi-class
                    try:
                        y_proba = self.predict(X_test, return_proba=True)
                        if y_proba.shape[1] > 1:
                            metrics["ROC AUC"] = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
        
        return metrics
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'kernel_size': ParamSpace.integer(min_val=3, max_val=5, default=3),  # Reduced max from 7 to 5
            'stride': ParamSpace.integer(min_val=1, max_val=2, default=1),  # Reduced max from 3 to 2
            'padding': ParamSpace.integer(min_val=0, max_val=2, default=1),
            'epochs': ParamSpace.integer(min_val=10, max_val=200, default=100),
            'learning_rate': ParamSpace.float_range(min_val=1e-5, max_val=1e-1, default=0.001),
            'batch_size': ParamSpace.integer(min_val=16, max_val=128, default=32),
            'weight_decay': ParamSpace.float_range(min_val=0.0, max_val=0.01, default=0.001),
            'optimizer': ParamSpace.categorical(choices=['AdamW', 'Adam', 'SGD'], default='AdamW'), # For SGD, momentum is fixed at 0.9
        }

    def set_params(self, **params):
        """Set parameters by recreating the model with new parameters"""
        self.create_model(**params)
        return self

    @property
    def is_initialized(self) -> bool:
        return hasattr(self, 'features') and hasattr(self, 'model')
    
    def _has_trained_weights(self) -> bool:
        """Check if the model has been trained (weights differ from initialization)."""
        if not self.is_initialized:
            return False
        
        # Check if any parameter has significantly non-zero values
        # (untrained models typically have small random values)
        for param in self.parameters():
            if param.abs().mean() > 0.1:  # Threshold for "trained" weights
                return True
        return False



import torch
import torch.nn as nn
from typing import Dict
from .ParamSpace import ParamSpace
from .base import BaseModel

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
    
    def train(self, X_train, y_train, epochs=100, learning_rate=0.001, batch_size=32):
        """Train the CNN model with PyTorch training loop."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call create_model() first.")
        
        # Convert to tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        
        # Create data loader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        self.train_mode = True  # Set to training mode
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % (epochs // 10) == 0:  # Print every 10% of training
                print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
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
            'learning_rate': ParamSpace.float(min_val=1e-5, max_val=1e-1, default=0.001, log_scale=True),
            'batch_size': ParamSpace.integer(min_val=16, max_val=128, default=32),
        }

    @property
    def is_initialized(self) -> bool:
        return hasattr(self, 'features') and hasattr(self, 'model')



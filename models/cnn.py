import torch
import torch.nn as nn
from typing import Dict
from .ParamSpace import ParamSpace

class CNNModel(nn.Module):   
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
    
    def get_param_space(self) -> Dict[str, ParamSpace]:
        return {
            'kernel_size': ParamSpace.integer(min_val=3, max_val=7, default=4),
            'stride': ParamSpace.integer(min_val=1, max_val=3, default=2),
            'padding': ParamSpace.integer(min_val=0, max_val=2, default=1)
        }

    @property
    def is_initialized(self) -> bool:
        return hasattr(self, 'features') and hasattr(self, 'model')



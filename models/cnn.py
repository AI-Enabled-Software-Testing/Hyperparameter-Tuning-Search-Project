import torch
import torch.nn as nn

class CNNModel(nn.Module):   
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv1: 32×32×1 -> 16×16×32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv2: 16×16×32 -> 8×8×64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv3: 8×8×64 -> 4×4×128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 4×4×128 -> 1×1×128
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 1×1×128 -> 128
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )
        
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    





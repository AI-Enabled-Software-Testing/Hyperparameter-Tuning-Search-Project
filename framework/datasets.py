import numpy as np
import torch
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = np.asarray(image).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)

        return image, label


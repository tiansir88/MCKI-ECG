import numpy as np
import torch
from torch.utils.data import Dataset

PTBXL5 = ["NORM", "MI", "STTC", "CD", "HYP"]

class ExternalNPYDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)   # (1000, 12)
        y = torch.tensor(self.y[idx], dtype=torch.float32)   # (5,)
        return x, y

import torch
from torch.utils.data import Dataset

class WiderFaceDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (list/array): Your input data.
            labels (list/array): Corresponding labels.
            transform (callable, optional): Optional transform to apply to data.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get data and label for a given index
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)  # Apply transform if provided
        
        # sample: (batch, 3, H, W)
        # coordinates: [batch, seq_len, 2]
        # heatmaps: ? [batch, seq_len, 1, H, W] (TYPICALLY GENERATED ON THE FLY)
        return sample, seq_len



import os
import torch
from torch.utils.data import Dataset
import pickle

class BasketballDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                with open(os.path.join(directory, filename), 'rb') as f:
                    self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert lists of player IDs to tensors
        home = torch.tensor(sample['home'])
        away = torch.tensor(sample['away'])
        # concat home and away
        lineup = torch.cat((home, away))

        # Convert plus_minus to tensor and add extra dimension (to make it a 1D tensor)
        plus_minus = torch.tensor([sample['plus_minus']])
        # Return the sample as a tuple
        return lineup, plus_minus
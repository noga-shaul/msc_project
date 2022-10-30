import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomSignalDataset(Dataset):
    def __init__(self, sigma=5, epoch_len=10000, transform=None, target_transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.signal_std = sigma
        self.epoch_len = epoch_len
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        #to do: return same sample for given index
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        sample = self.signal_std * torch.randn(1)
        label = sample
        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return sample, label


def load_data(sigma=5, epoch_len=10000, test_len=1000):
    train_data = CustomSignalDataset(sigma=sigma, epoch_len=epoch_len)
    test_data = CustomSignalDataset(sigma=sigma, epoch_len=test_len)
    #train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    #test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_data, test_data

#train_data = CustomSignalDataset()
#train_loader, test_loader = load_data()
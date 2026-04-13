import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter


def bandpass_filter(signal, low=1, high=40, fs=128, order=5):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq

    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)


def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)


class EEGDataset(Dataset):
    def __init__(self, x, y, fs=256):
        self.x = x
        self.y = y
        self.fs = fs

    def preprocess(self, signal):
        processed = np.zeros_like(signal)

        for ch in range(signal.shape[0]):
            x = signal[ch]

            x = bandpass_filter(x, fs=self.fs)
            x = normalize(x)

            processed[ch] = x

        return processed

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        x = self.preprocess(x)

        y = torch.tensor(self.y[idx]).squeeze().long()

        return torch.tensor(x, dtype=torch.float32), y




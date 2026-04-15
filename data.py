import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder


def bandpass_filter(signal, low=1, high=40, fs=250, order=5):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


class EEGDataset(Dataset):
    def __init__(self, x, y, fs=250):
        self.fs = fs

        self.x = self.preprocess_all(x)

        # labels -> tensor
        self.y = torch.tensor(y, dtype=torch.long)

    def preprocess_all(self, x):
        processed = []

        for sample in x:
            out = np.zeros_like(sample)

            for ch in range(sample.shape[0]):
                sig = sample[ch]
                sig = bandpass_filter(sig, fs=self.fs)
                out[ch] = sig

            processed.append(out)

        return torch.tensor(np.array(processed), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
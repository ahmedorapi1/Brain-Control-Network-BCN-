import os
from glob import glob
import numpy as np
import math
import onnxruntime as ort
import mne
from scipy.interpolate import griddata
from matplotlib import colormaps
import torch
import random
import time
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
device = torch.device('cpu')


#ده المجلد مش الفايلات يابو صلاح
DATA_PATH = r"D:\GP_data\EEG\sample"

class ATCNet_LSTM(nn.Module):
    def __init__(self, n_channels=44, n_times=641,
                 num_classes=4, hidden=256):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, (n_channels, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.25)
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(8),
            nn.Dropout(0.25),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.25)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            dropout=0.25,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(64)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        B = x.size(0)

        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.spatial(x)  # (B, 16, 1, T)
        x = x.squeeze(2)  # (B, 16, T)

        x = self.temporal(x)  # (B, 64, T')

        x = x.permute(0, 2, 1)  # (B, T', 64)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)  # residual connection

        out, _ = self.lstm(x)  # (B, T', hidden*2)
        out = out[:, -1, :]  # آخر timestep

        return self.classifier(out)


#   0 = Left fist   (R03,R07,R11 → T1)
#   1 = Right fist  (R03,R07,R11 → T2)
#   2 = Both fists  (R04,R08,R12 → T1)
#   3 = Both feet   (R04,R08,R12 → T2)
class EEGDatasetBuilder:
    def __init__(self, data_path, target_sfreq=160):
        self.data_path = data_path
        self.target_sfreq = target_sfreq
        self.target_runs = ['R03', 'R04', 'R07', 'R08', 'R11', 'R12']
        self._last_ch_names = None

        self.target_channels_clean = [
            "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
            "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
            "T7", "T8",
            "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
            "O1", "OZ", "O2"
        ]

    def load_subject(self):
        all_files = sorted(glob(
            os.path.join(self.data_path, "*", "*.edf")
        ))
        filtered = [
            f for f in all_files
            if any(run in os.path.basename(f) for run in self.target_runs)
        ]
        print(f"Total files: {len(all_files)}  |  After run filter: {len(filtered)}")
        return filtered

    def load_files(self):
        all_files = sorted(glob(
            os.path.join(self.data_path, "*.edf")
        ))
        filtered = [
            f for f in all_files
            if any(run in os.path.basename(f) for run in self.target_runs)
        ]
        print(f"Total files: {len(all_files)}  |  After run filter: {len(filtered)}")
        return filtered

    @staticmethod
    def _clean_ch_names(raw):
        rename_dict = {ch: ch.rstrip('.').upper() for ch in raw.ch_names}
        raw.rename_channels(rename_dict)
        return raw

    @staticmethod
    def _annotation_to_label(description, basename):
        is_fist_run = any(r in basename for r in ['R03', 'R07', 'R11'])

        if description == 'T0':
            return -1
        elif description == 'T1':
            return 0 if is_fist_run else 2
        elif description == 'T2':
            return 1 if is_fist_run else 3
        return -1

    def get_subjects(self):
        files = self.load_files()
        subjects = sorted(list({
            os.path.basename(f).split("R")[0]
            for f in files}))
        return subjects

    def build(self):
        X, y = [], []
        skipped = 0
        last_raw = None

        edf_files = self.load_files()

        subjects = []
        for file in edf_files:
            subject_id = os.path.basename(file).split("R")[0]
            try:
                raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
                raw = self._clean_ch_names(raw)

                if raw.info['sfreq'] != self.target_sfreq:
                    raw.resample(self.target_sfreq, verbose=False)

                available = [ch for ch in self.target_channels_clean
                             if ch in raw.ch_names]
                if len(available) == 0:
                    skipped += 1
                    continue
                raw.pick(available, verbose=False)

                raw.filter(4., 35., verbose=False)

                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(events) == 0:
                    skipped += 1
                    continue

                tmax = 4.0
                epochs = mne.Epochs(
                    raw, events, event_id=event_id,
                    tmin=0.0, tmax=tmax,
                    baseline=None, preload=True,
                    reject_by_annotation=False,
                    verbose=False
                )

                data = epochs.get_data()
                raw_labels = epochs.events[:, -1]
                descriptions = [
                    list(event_id.keys())[list(event_id.values()).index(l)]
                    for l in raw_labels
                ]

                basename = os.path.basename(file)
                labels = np.array([
                    self._annotation_to_label(d, basename)
                    for d in descriptions
                ])

                valid = labels >= 0
                data = data[valid]
                labels = labels[valid]

                if len(data) == 0:
                    skipped += 1
                    continue

                expected_T = int(tmax * self.target_sfreq) + 1
                if data.shape[2] > expected_T:
                    data = data[:, :, :expected_T]
                elif data.shape[2] < expected_T:
                    pad = expected_T - data.shape[2]
                    data = np.pad(data, ((0, 0), (0, 0), (0, pad)))

                X.append(data.astype(np.float32))
                y.append(labels)
                last_raw = raw
                subjects.append(np.array([subject_id] * len(labels)))

            except Exception as e:
                print(f"Skip: {os.path.basename(file)}  →  {e}")
                skipped += 1
                continue

        print(f"Skipped: {skipped} / {len(edf_files)}")

        if len(X) == 0:
            raise ValueError("No valid epochs found")

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        self._last_ch_names = list(last_raw.ch_names)

        print(f"X shape  : {X.shape}  dtype: {X.dtype}")
        print(f"y shape  : {y.shape}  classes: {np.unique(y)}")
        print(f"Per class: {np.bincount(y)}")
        subjects = np.concatenate(subjects)
        return X, y, self._last_ch_names, subjects

def encode_labels(y):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y).astype(np.int64)
    print(f"Classes: {le.classes_}  →  {np.unique(y_enc)}")
    return y_enc, le

class EEGTorchDataset(Dataset):
    """
    X: (N, C, T)  float32
    y: (N,)       int64
    Output: (C, T) tensor مباشرة للموديل
    """
    def __init__(self, X, y):
        mean = X.mean(axis=(1, 2), keepdims=True)
        std  = X.std(axis=(1, 2),  keepdims=True) + 1e-8
        self.X = ((X - mean) / std).astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx])
        )

def make_weighted_sampler(y_train):
    class_counts  = np.bincount(y_train)
    sample_weights = np.array([1.0 / class_counts[y] for y in y_train])
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float32),
        num_samples = len(sample_weights),
        replacement = True
    )
    print(f"Class counts: {class_counts}")
    return sampler


builder = EEGDatasetBuilder(DATA_PATH)
X, y_raw, ch_names, subjects = builder.build()

y, label_encoder = encode_labels(y_raw)
num_classes = len(label_encoder.classes_)
class_counts = np.bincount(y)


ds = EEGTorchDataset(X, y)

pin = torch.cuda.is_available()
sampler = make_weighted_sampler(y)

loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=0, pin_memory=pin)
model = ATCNet_LSTM(n_channels=44, num_classes=4)

# load ONNX model
onnx_path = r"D:\GP_data\EEG\our paper\LSTM+finetuning\MI.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# خد sample واحدة من الـ loader
X_sample, y_sample = next(iter(loader))

# اختار أول sample فقط
x = X_sample[0:1].numpy().astype(np.float32)   # shape: (1, ...)

true_label = y_sample[0].item()

# inference
output = session.run([output_name], {input_name: x})[0]

pred_label = np.argmax(output, axis=1)[0]

# النتيجة
print("True Label:", true_label)
print("Pred Label:", pred_label)

if pred_label == true_label:
    print("✅ Correct prediction")
else:
    print("❌ Wrong prediction")
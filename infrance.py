import os
import sys
import argparse

import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from matplotlib import colormaps


class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2)
        )
        self.drop = nn.Dropout(0.15)
        self.fc = nn.Linear(64 * 8 * 8, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.drop(self.fc(x))
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=4, hidden=128):
        super().__init__()
        self.cnn = FrameCNN()
        self.lstm = nn.LSTM(input_size=128,
            hidden_size=hidden,
            batch_first=True
                           )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.cnn(x)
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        out = out[:, -1, :]
        return self.classifier(out)

# ─────────────────────────────────────────────
# 2.  PREPROCESSING  (mirrors training pipeline)
# ─────────────────────────────────────────────

TARGET_SFREQ = 160
TARGET_TMAX  = 4.0
WIN_SIZE     = 256
STEP         = 16

TARGET_CHANNELS = [
    "FC5","FC3","FC1","FCZ","FC2","FC4","FC6",
    "C5","C3","C1","CZ","C2","C4","C6",
    "CP5","CP3","CP1","CPZ","CP2","CP4","CP6",
    "F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "T7","T8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "O1","OZ","O2",
]

LABEL_NAMES = ["Left fist", "Right fist", "Both fists", "Both feet"]


def clean_ch_names(raw):
    rename = {ch: ch.rstrip('.').upper() for ch in raw.ch_names}
    raw.rename_channels(rename)
    return raw


def annotation_to_label(description, is_fist_run):
    if description == 'T0':
        return -1
    elif description == 'T1':
        return 0 if is_fist_run else 2
    elif description == 'T2':
        return 1 if is_fist_run else 3
    return -1


def load_edf(edf_path, run_type=None):
    basename = os.path.basename(edf_path)

    # Determine run type
    if run_type == 'fist':
        is_fist_run = True
    elif run_type == 'limb':
        is_fist_run = False
    else:
        # Auto-detect from filename (e.g. S001R03.edf)
        is_fist_run = any(r in basename for r in ['R03', 'R07', 'R11'])
        if not is_fist_run and not any(r in basename for r in ['R04', 'R08', 'R12']):
            print(
                f"[WARN] Could not auto-detect run type from '{basename}'.\n"
                "       Assuming 'fist' run (left/right). "
                "Use --run_type fist|limb to override."
            )
            is_fist_run = True

    print(f"[INFO] Loading  : {basename}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw = clean_ch_names(raw)

    # Resample
    if raw.info['sfreq'] != TARGET_SFREQ:
        print(f"[INFO] Resampling {raw.info['sfreq']} Hz → {TARGET_SFREQ} Hz")
        raw.resample(TARGET_SFREQ, verbose=False)

    # Channel selection
    available = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
    if len(available) == 0:
        raise ValueError(
            f"No target channels found in {basename}.\n"
            f"File channels: {raw.ch_names}"
        )
    print(f"[INFO] Channels : {len(available)} / {len(TARGET_CHANNELS)} matched")
    raw.pick(available, verbose=False)

    # Band-pass filter
    raw.filter(1.0, 40.0, verbose=False)

    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if len(events) == 0:
        raise ValueError(f"No annotations/events found in {basename}.")

    # Epoch
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=0.0, tmax=TARGET_TMAX,
        baseline=None, preload=True,
        reject_by_annotation=False,
        verbose=False,
    )

    data = epochs.get_data()               # (N, C, T)
    raw_labels = epochs.events[:, -1]
    descriptions = [
        list(event_id.keys())[list(event_id.values()).index(l)]
        for l in raw_labels
    ]
    labels = np.array([
        annotation_to_label(d, is_fist_run) for d in descriptions
    ])

    # Pad / truncate to fixed length
    expected_T = int(TARGET_TMAX * TARGET_SFREQ) + 1
    if data.shape[2] > expected_T:
        data = data[:, :, :expected_T]
    elif data.shape[2] < expected_T:
        pad  = expected_T - data.shape[2]
        data = np.pad(data, ((0, 0), (0, 0), (0, pad)))

    print(f"[INFO] Epochs   : {len(data)} total  "
          f"({(labels >= 0).sum()} task, {(labels < 0).sum()} rest/skip)")

    return data.astype(np.float32), labels, list(raw.ch_names)


# ─────────────────────────────────────────────
# 3.  TOPOMAP CONVERSION  (mirrors training)
# ─────────────────────────────────────────────

def get_electrode_positions(ch_names):
    montage   = mne.channels.make_standard_montage("standard_1020")
    pos       = montage.get_positions()['ch_pos']
    pos_upper = {k.upper(): v for k, v in pos.items()}

    matched_ch     = []
    matched_coords = []
    for ch in ch_names:
        if ch.upper() in pos_upper:
            matched_ch.append(ch)
            matched_coords.append(pos_upper[ch.upper()])

    if len(matched_ch) == 0:
        raise ValueError("No channels matched montage positions.")

    coords = np.array(matched_coords)
    x_sel  = coords[:, 0]
    y_sel  = coords[:, 1]

    x_norm = (x_sel - x_sel.min()) / (x_sel.max() - x_sel.min() + 1e-8) * 2 - 1
    y_norm = (y_sel - y_sel.min()) / (y_sel.max() - y_sel.min() + 1e-8) * 2 - 1

    xi, yi = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    return x_norm, y_norm, xi, yi, matched_ch


def eeg_to_topomap(epoch, x_sel, y_sel, xi, yi):
    """
    Convert one epoch (C, T) → tensor (3, T_win, 32, 32).
    """
    maps = []
    for i in range(0, epoch.shape[1] - WIN_SIZE, STEP):
        seg = np.mean(epoch[:, i:i + WIN_SIZE], axis=1)
        zi  = griddata((x_sel, y_sel), seg, (xi, yi), method="cubic")
        zi  = np.nan_to_num(zi, nan=0.0)
        zi  = (zi - zi.min()) / (zi.max() - zi.min() + 1e-8)
        rgb = colormaps["jet"](zi)[:, :, :3]
        maps.append(rgb.astype(np.float32))

    if len(maps) == 0:
        raise ValueError(
            f"Epoch too short (T={epoch.shape[1]}) for win_size={WIN_SIZE}. "
            f"Minimum required: {WIN_SIZE + 1} samples."
        )

    # (T_win, 32, 32, 3)  →  (3, T_win, 32, 32)
    stacked = np.stack(maps, axis=0)                   # (T_win, 32, 32, 3)
    stacked = np.transpose(stacked, (3, 0, 1, 2))      # (3, T_win, 32, 32)
    return stacked


def load_model(model_path, device):
    print(f"Loading model: {model_path}")

    model = CNN_LSTM(num_classes=4)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    label_classes = [0, 1, 2, 3]
    matched_ch = TARGET_CHANNELS

    return model, matched_ch, label_classes


def predict(model_path, edf_path, run_type=None, output_csv=None, skip_rest=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device   : {device}")

    # Load model & checkpoint metadata
    model, train_ch, label_classes = load_model(model_path, device)

    # Load & preprocess EDF
    data, labels, ch_names = load_edf(edf_path, run_type=run_type)

    # Align channels to those used during training
    # (use intersection in training order)
    ch_upper     = {ch.upper(): ch for ch in ch_names}
    aligned_ch   = [ch for ch in train_ch if ch.upper() in ch_upper]
    if len(aligned_ch) == 0:
        raise ValueError(
            "No overlap between EDF channels and model training channels.\n"
            f"EDF      : {ch_names}\n"
            f"Training : {train_ch}"
        )
    if len(aligned_ch) < len(train_ch):
        missing = set(train_ch) - set(aligned_ch)
        print(f"[WARN] {len(missing)} training channels missing from EDF: {missing}")

    ch_idx = [ch_names.index(ch_upper[ch.upper()]) for ch in aligned_ch]
    data   = data[:, ch_idx, :]   # select & reorder

    # Electrode positions for the aligned channels
    x_sel, y_sel, xi, yi, matched_ch = get_electrode_positions(aligned_ch)
    ch_idx2 = [aligned_ch.index(ch) for ch in matched_ch]
    data    = data[:, ch_idx2, :]

    print(f"\n{'─'*55}")
    print(f"{'Epoch':>6}  {'True Label':>12}  {'Prediction':>12}  {'Confidence':>10}")
    print(f"{'─'*55}")

    results = []
    for i, (epoch, true_label) in enumerate(zip(data, labels)):
        if skip_rest and true_label < 0:
            continue

        # Build topomap tensor: (3, T_win, 32, 32) → add batch → (1, 3, T_win, 32, 32)
        try:
            topo = eeg_to_topomap(epoch, x_sel, y_sel, xi, yi)
        except ValueError as e:
            print(f"[WARN] Epoch {i}: {e}  (skipped)")
            continue

        x_tensor = torch.tensor(topo, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_tensor)               # (1, num_classes)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx    = int(np.argmax(probs))
        confidence  = float(probs[pred_idx])

        pred_name  = LABEL_NAMES[label_classes[pred_idx]] \
                     if pred_idx < len(label_classes) else f"Class {pred_idx}"
        true_name  = LABEL_NAMES[true_label] if 0 <= true_label < len(LABEL_NAMES) \
                     else ("Rest" if true_label < 0 else f"Class {true_label}")

        print(f"{i:>6}  {true_name:>12}  {pred_name:>12}  {confidence*100:>9.1f}%")

        results.append({
            "epoch":      i,
            "true_label": true_label,
            "true_name":  true_name,
            "pred_idx":   pred_idx,
            "pred_name":  pred_name,
            "confidence": round(confidence, 4),
            **{f"prob_{LABEL_NAMES[c]}": round(float(probs[j]), 4)
               for j, c in enumerate(label_classes)},
        })

    print(f"{'─'*55}")

    # Summary
    task_results = [r for r in results if r["true_label"] >= 0]
    if task_results:
        correct = sum(r["true_name"] == r["pred_name"] for r in task_results)
        print(f"\n[INFO] Task epochs : {len(task_results)}")
        print(f"[INFO] Accuracy    : {correct}/{len(task_results)} "
              f"= {correct/len(task_results)*100:.1f}%")

    # Optional CSV export
    if output_csv:
        import csv
        if results:
            fieldnames = list(results[0].keys())
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"[INFO] Results saved → {output_csv}")

    return results

if __name__ == "__main__":
    predict(
        model_path="model.pth",
        edf_path="sample.edf"
    )
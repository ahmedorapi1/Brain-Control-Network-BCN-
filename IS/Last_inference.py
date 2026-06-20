# !pip install onnxruntime

import numpy as np
import onnxruntime as rt
from scipy.interpolate import griddata
import matplotlib.cm as cm
import os

# --- SPATIAL CONFIGURATIONS (Required for griddata) ---
map_size = 32
start_map = 6
end_map = 44

# Channel labels and mapping used during segments generation
label_to_channel = {
    'Fp1': 1, 'Fp2': 2, 'F7': 3, 'F3': 4, 'Fz': 5, 'F4': 6, 'F8': 7, 'FC5': 8, 'FC1': 9, 'FC2': 10, 'FC6': 11,
    'T7': 12, 'C3': 13, 'Cz': 14, 'C4': 15, 'T8': 16, 'TP9': 17, 'CP5': 18, 'CP1': 19, 'CP2': 20, 'CP6': 21,
    'TP10': 22, 'P7': 23, 'P3': 24, 'Pz': 25, 'P4': 26, 'P8': 27, 'PO9': 28, 'O1': 29, 'Oz': 30, 'O2': 31,
    'PO10': 32, 'AF7': 33, 'AF3': 34, 'AF4': 35, 'AF8': 36, 'F5': 37, 'F1': 38, 'F2': 39, 'F6': 40, 'FT9': 41,
    'FT7': 42, 'FC3': 43, 'FC4': 44, 'FT8': 45, 'FT10': 46, 'C5': 47, 'C1': 48, 'C2': 49, 'C6': 50, 'TP7': 51,
    'CP3': 52, 'CPz': 53, 'CP4': 54, 'TP8': 55, 'P5': 56, 'P1': 57, 'P2': 58, 'P6': 59, 'PO7': 60, 'PO3': 61,
    'POz': 62, 'PO4': 63, 'PO8': 64
}
Channels4 = ["F7","F3","FC5","FC6","F4","T7","T8","C3","C4","P7","P8","O1","O2","CP5","CP6","Cz","P3","P4","POz"]
selected_channels = np.array([label_to_channel[ch] - 1 for ch in Channels4])

ALL_MNT_X = np.array([-0.34, 0.34, -0.81, -0.47, 0.0, 0.47, 0.81, -0.68, -0.24, 0.24, 0.68, -0.99, -0.54, 0.0, 0.54, 0.99, -0.87, -0.44, 0.44, 0.87, -0.65, 0.65, -0.81, -0.47, 0.0, 0.47, 0.81, -0.54, -0.54, 0.0, 0.54, 0.54, -0.54, -0.27, 0.27, 0.54, -0.68, -0.31, 0.31, 0.68, -0.92, -0.74, -0.37, 0.37, 0.74, 0.92, -0.74, -0.27, 0.27, 0.74, -0.87, -0.44, 0.0, 0.44, 0.87, -0.65, -0.27, 0.27, 0.65, -0.54, -0.27, 0.0, 0.27, 0.54])
ALL_MNT_Y = np.array([0.87, 0.87, 0.34, 0.47, 0.57, 0.47, 0.34, 0.13, 0.21, 0.21, 0.13, -0.06, -0.07, -0.08, -0.07, -0.06, -0.26, -0.26, -0.26, -0.26, -0.41, -0.41, -0.47, -0.62, -0.67, -0.62, -0.47, -0.74, -0.74, -0.78, -0.74, -0.74, 0.71, 0.76, 0.76, 0.71, 0.52, 0.56, 0.56, 0.52, 0.21, 0.25, 0.28, 0.28, 0.25, 0.21, -0.11, -0.11, -0.11, -0.11, -0.31, -0.31, -0.31, -0.31, -0.31, -0.52, -0.54, -0.54, -0.52, -0.71, -0.74, -0.78, -0.74, -0.71])

mnt_x = ALL_MNT_X[selected_channels]
mnt_y = ALL_MNT_Y[selected_channels]

xi, yi = np.meshgrid(
    np.linspace(np.min(ALL_MNT_X), np.max(ALL_MNT_X), map_size),
    np.linspace(np.min(ALL_MNT_Y), np.max(ALL_MNT_Y), map_size)
)

def generate_topomaps_from_segments(segments):
    n_maps_local = segments.shape[0]
    trial_maps = np.zeros((map_size, map_size, 3, n_maps_local))

    for w in range(n_maps_local):
        segment = segments[w, :]

        # Interpolation using local coordinate definitions
        zi = griddata((mnt_x, mnt_y), segment, (xi, yi), method='cubic')
        zi = np.nan_to_num(zi, nan=0.0)

        # Normalization for colormap
        zi_min, zi_max = np.min(zi), np.max(zi)
        zi_norm = (zi - zi_min) / (zi_max - zi_min + np.finfo(float).eps)

        # Apply Colormap
        cmap = cm.get_cmap('jet')
        trial_maps[:, :, :, w] = cmap((zi_norm * 255).astype(np.uint8))[:, :, :3]

    # Slice to the specific window range used during training
    trial_maps = trial_maps[:, :, :, start_map-1:end_map]
    return np.expand_dims(trial_maps, axis=0).astype(np.float32)

def main_from_segments():
    model_path = "speech_model.onnx"
    classes = ["Hello", "Help_me", "Stop"]

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    for class_name in classes:
        seg_file = f"segments_{class_name}.npy"
        if not os.path.exists(seg_file):
            print(f"Warning: {seg_file} not found. Skipping...")
            continue

        segments = np.load(seg_file)
        processed_input = generate_topomaps_from_segments(segments)

        preds = session.run(None, {input_name: processed_input})[0]
        pred_class = np.argmax(preds, axis=1)[0]
        
        print(f"Class: {class_name:<8} | Probs: {np.round(preds[0], 4)} | Predicted Index: {pred_class}")

if __name__ == "__main__":
    main_from_segments()
from data import EEGDataset
from torch.utils.data import DataLoader
import torch
from EEGNet import EEG_MODEL
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix

test_data_path = r"C:\Users\hp\Desktop\EEG\BCI2020 EEG Signal for Words\Validation set\data_sample01.mat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = EEG_MODEL()
model.load_state_dict(torch.load("eeg-net.pth"))
model.to(device)
model.eval()


def get_data(path):
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    epo = data["epo_validation"]
    x = epo.x
    y = epo.y
    y = np.argmax(y, axis=0)
    x = np.transpose(x, (2, 1, 0))
    x = np.expand_dims(x, 1)

    return x, y



x_test, y_test = get_data(test_data_path)
test_dataset = EEGDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def cm(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix", fontsize=14)

    plt.colorbar()

    classes = np.unique(y_true)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12
            )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        _, pred = torch.max(outputs, 1)

        total += y_batch.size(0)
        correct += (pred == y_batch).sum().item()

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print("Accuracy:", correct / total)

cm(all_labels, all_preds)



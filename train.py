import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from EEGNet import EEG_MODEL
from data import EEGDataset

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEG_MODEL().to(device)
# model.load_state_dict(torch.load("best_model.pth"))
# model.to(device)

def get_data():
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)

    x, y, metadata = paradigm.get_data(dataset=dataset)

    x = np.expand_dims(x, 1)

    le = LabelEncoder()
    y = le.fit_transform(y)

    train_idx = metadata['session'] == '0train'
    test_idx = metadata['session'] == '1test'

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_data()


train_dataset = EEGDataset(x_train, y_train)
test_dataset = EEGDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)


def evaluate(loader):
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            _, pred = torch.max(outputs, 1)

            total += y_batch.size(0)
            correct += (pred == y_batch).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return correct / total, all_labels, all_preds


best_acc = 0


for epoch in range(150):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)


        optimizer.zero_grad()

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    acc, _, _ = evaluate(test_loader)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch}: Loss = {total_loss: .4f}, Test Acc = {acc: .4f}")


print("Best Accuracy:", best_acc)


def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")

    classes = np.unique(y_true)
    ticks = np.arange(len(classes))

    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


acc, y_true, y_pred = evaluate(test_loader)

print("Final Accuracy:", acc)
plot_cm(y_true, y_pred)

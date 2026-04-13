import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from EEGNet import EEG_MODEL
import numpy as np
from data import EEGDataset
import scipy.io as sio
train_data_path = r"C:\Users\hp\Desktop\EEG\BCI2020 EEG Signal for Words\Training set\data_sample01.mat"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEG_MODEL().to(device)


def get_data(path):
    data = sio.loadmat(path)
    epo = data["epo_train"][0][0]
    x = epo[4]
    y = epo[5]   # (5, 300)
    y = np.argmax(y, axis=0)  # (300,)

    x = np.transpose(x, (2, 1, 0))
    x = np.expand_dims(x, 1)
    return x, y


x_train, y_train = get_data(train_data_path)
train_dataset = EEGDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

model.train()
for epoch in range(60):
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

    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
torch.save(model.state_dict(), "eegnet.pth")
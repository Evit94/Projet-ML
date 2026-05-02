import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from preprocessing import X_train, X_test, Y_train, Y_test

# Conversion numpy → tenseurs PyTorch
# CIFAR attend (N, C, H, W) donc on reshape (50000, 3, 32, 32)
X_train_t = torch.tensor(X_train.reshape(-1, 3, 32, 32), dtype=torch.float32)
X_test_t  = torch.tensor(X_test.reshape(-1, 3, 32, 32), dtype=torch.float32)
Y_train_t = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long)
Y_test_t  = torch.tensor(np.argmax(Y_test, axis=1), dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train_t, Y_train_t)
test_dataset  = TensorDataset(X_test_t, Y_test_t)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=64)

# Architecture CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc    = nn.Linear(4096, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # (N, 64, 32, 32)
        x = torch.relu(self.conv2(x))   # (N, 64, 32, 32)
        x = self.pool(x)                # (N, 64, 16, 16)
        x = torch.relu(self.conv3(x))   # (N, 64, 16, 16)
        x = self.pool(x)                # (N, 64, 8, 8)
        x = torch.relu(self.conv4(x))   # (N, 64, 8, 8)
        x = torch.flatten(x, 1)         # (N, 4096)
        x = self.fc(x)                  # (N, 10)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement
def train(epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, loss = {total_loss/len(train_loader):.4f}")

# Évaluation
def evaluate(loader):
    errors = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            errors += (preds != Y_batch).sum().item()
            total += len(Y_batch)
    return (errors / total) * 100

train(epochs=10)
print(f"Taux d'erreur train : {evaluate(train_loader):.2f}%")
print(f"Taux d'erreur test  : {evaluate(test_loader):.2f}%")
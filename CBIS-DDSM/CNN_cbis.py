#CNN CBIS-DDSM - classification binaire bénin / malin
#Style proche du CNN CIFAR-10 : PyTorch, fonctions simples, Adam, early stopping.

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from preprocessing import (X_train, y_train, X_val, y_val, X_test, y_test,
                           pos_weight, afficher_repartition)

afficher_repartition()

# Conversion numpy -> tenseurs PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

# Architecture : 3 blocs conv + pooling, puis 2 couches denses
# Entrée : (N, 1, 128, 128)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 16 * 16, 128)
        self.fc2   = nn.Linear(128, 1)  # 1 sortie (logit) pour la classification binaire

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # (N, 32, 64, 64)
        x = self.pool(torch.relu(self.conv2(x)))  # (N, 64, 32, 32)
        x = self.pool(torch.relu(self.conv3(x)))  # (N, 64, 16, 16)
        x = torch.flatten(x, 1)                    # (N, 16384)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                            # (N, 1) logit
        return x

model = CNN()
# Weighted loss pour gérer le déséquilibre des classes
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
optimizer = optim.Adam(model.parameters(), lr=0.001)

os.makedirs("CBIS-DDSM/params", exist_ok=True)
os.makedirs("results/cbis_ddsm", exist_ok=True)


def perte_validation(loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            total += criterion(model(X_batch), Y_batch).item()
    model.train()
    return total / len(loader)


# Entraînement avec early stopping (sur la loss de validation) + sauvegarde du meilleur modèle
def train(epochs=20, patience=5):
    train_loss_hist = []
    val_loss_hist = []
    best_val = float("inf")
    sans_amelioration = 0

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        tl = total_loss / len(train_loader)
        vl = perte_validation(val_loader)
        train_loss_hist.append(tl)
        val_loss_hist.append(vl)
        print(f"Epoch {epoch+1}, loss train = {tl:.4f}, loss val = {vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), "CBIS-DDSM/params/cnn_cbis_best.pth")
            sans_amelioration = 0
        else:
            sans_amelioration += 1
            if sans_amelioration >= patience:
                print(f"Early stopping à l'epoch {epoch+1}")
                break

    return train_loss_hist, val_loss_hist


# Prédictions (probabilités et classes) sur un loader
def predire(loader):
    model.eval()
    probas = []
    vrais = []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            logits = model(X_batch)
            p = torch.sigmoid(logits)
            probas.extend(p.numpy().flatten())
            vrais.extend(Y_batch.numpy().flatten())
    model.train()
    return np.array(vrais), np.array(probas)


train_loss_hist, val_loss_hist = train(epochs=20, patience=5)

# On recharge le meilleur modèle
model.load_state_dict(torch.load("CBIS-DDSM/params/cnn_cbis_best.pth"))

# Évaluation finale sur le test
y_true, y_proba = predire(test_loader)
y_pred = (y_proba >= 0.5).astype(int)

metrics = {
    "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
    "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
}

print("\n=== Métriques (test) ===")
for nom, val in metrics.items():
    print(f"{nom:<10} : {val}")

# Sauvegarde des métriques + historique (réutilisé par analyse_cbis.py)
sortie = {
    "metrics": metrics,
    "train_loss": train_loss_hist,
    "val_loss": val_loss_hist,
    "y_true": y_true.tolist(),
    "y_proba": y_proba.tolist(),
}
with open("results/cbis_ddsm/metrics.json", "w") as f:
    json.dump(sortie, f, indent=2)
print("\nMétriques sauvegardées : results/cbis_ddsm/metrics.json")

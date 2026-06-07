import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import X_train, X_test, Y_train, Y_test

# Conversion numpy → tenseurs PyTorch
# CIFAR attend (N, C, H, W) donc on reshape (50000, 3, 32, 32)
# Sur CPU l'entraînement complet est très long, on prend donc un sous-ensemble
N_TRAIN = 15000   # taille du sous-ensemble d'apprentissage (train + validation)
N_VAL   = 3000    # validation découpée DEPUIS le train (pas depuis le test)
N_TEST  = 5000

X_full = torch.tensor(X_train[:N_TRAIN].reshape(-1, 3, 32, 32), dtype=torch.float32)
Y_full = torch.tensor(np.argmax(Y_train[:N_TRAIN], axis=1), dtype=torch.long)

# Split train / validation. Le TEST n'est utilisé qu'une seule fois, à la fin,
# pour ne pas biaiser l'early stopping ni le choix du meilleur modèle.
X_train_t, Y_train_t = X_full[:N_TRAIN - N_VAL], Y_full[:N_TRAIN - N_VAL]
X_val_t,   Y_val_t   = X_full[N_TRAIN - N_VAL:], Y_full[N_TRAIN - N_VAL:]
X_test_t = torch.tensor(X_test[:N_TEST].reshape(-1, 3, 32, 32), dtype=torch.float32)
Y_test_t = torch.tensor(np.argmax(Y_test[:N_TEST], axis=1), dtype=torch.long)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=64)
test_loader  = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=64)

# Architecture CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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

os.makedirs("CIFAR-10/params", exist_ok=True)
os.makedirs("results/cifar10", exist_ok=True)

# Accuracy sur un loader (sert pour la courbe d'accuracy)
def accuracy(loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == Y_batch).sum().item()
            total += len(Y_batch)
    model.train()
    return (correct / total) * 100

# Entraînement avec early stopping + sauvegarde du meilleur modèle
# L'early stopping et le choix du meilleur modèle se font sur la VALIDATION.
def train(epochs=15, patience=3):
    loss_history = []
    acc_train_history = []
    acc_val_history = []
    best_acc = 0
    epochs_sans_amelioration = 0

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_moy = total_loss / len(train_loader)
        acc_tr = accuracy(train_loader)
        acc_vl = accuracy(val_loader)
        loss_history.append(loss_moy)
        acc_train_history.append(acc_tr)
        acc_val_history.append(acc_vl)
        print(f"Epoch {epoch+1}, loss = {loss_moy:.4f}, acc train = {acc_tr:.2f}%, acc val = {acc_vl:.2f}%")

        # Sauvegarde du meilleur modèle (selon l'accuracy de validation)
        if acc_vl > best_acc:
            best_acc = acc_vl
            torch.save(model.state_dict(), "CIFAR-10/params/cnn_best.pth")
            epochs_sans_amelioration = 0
        else:
            epochs_sans_amelioration += 1
            # Early stopping : on arrête si pas d'amélioration depuis 'patience' epochs
            if epochs_sans_amelioration >= patience:
                print(f"Early stopping à l'epoch {epoch+1}")
                break

    return loss_history, acc_train_history, acc_val_history

# Évaluation (taux d'erreur)
def evaluate(loader):
    errors = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            errors += (preds != Y_batch).sum().item()
            total += len(Y_batch)
    model.train()
    return (errors / total) * 100

# Matrice de confusion sur le test
def matrice_confusion(loader):
    matrice = np.zeros((10, 10), dtype=int)
    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            for vrai, pred in zip(Y_batch, preds):
                matrice[vrai.item(), pred.item()] += 1
    model.train()
    return matrice


loss_history, acc_train_history, acc_val_history = train(epochs=15, patience=3)

# On recharge le meilleur modèle (sélectionné sur la validation) pour
# l'évaluation finale sur le TEST, qui n'a servi à rien jusqu'ici.
model.load_state_dict(torch.load("CIFAR-10/params/cnn_best.pth"))

err_train = evaluate(train_loader)
err_test = evaluate(test_loader)
print(f"Taux d'erreur train : {err_train:.2f}%")
print(f"Taux d'erreur test  : {err_test:.2f}%")

# Sauvegarde des métriques du CNN (réutilisé par analyse_cifar.py)
metrics_cnn = {
    "acc_train": round(100 - err_train, 2),
    "acc_test": round(100 - err_test, 2),
    "err_train": round(err_train, 2),
    "err_test": round(err_test, 2),
}
with open("results/cifar10/metrics_cnn.json", "w") as f:
    json.dump(metrics_cnn, f, indent=2)

# Courbe de loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, "o-")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN CIFAR-10 - Courbe de loss")
plt.grid(True, ls=":")
plt.tight_layout()
plt.savefig("results/cifar10/loss_curve.png")
plt.close()

# Courbe d'accuracy (train et test)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(acc_train_history) + 1), acc_train_history, "o-", label="train")
plt.plot(range(1, len(acc_val_history) + 1), acc_val_history, "o-", label="validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN CIFAR-10 - Courbe d'accuracy")
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()
plt.savefig("results/cifar10/accuracy_curve.png")
plt.close()

# Matrice de confusion
classes = ["avion", "auto", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]
mc = matrice_confusion(test_loader)
plt.figure(figsize=(8, 6))
plt.imshow(mc, cmap="Blues")
plt.colorbar()
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.title("Matrice de confusion - CNN CIFAR-10")
plt.xticks(range(10), classes, rotation=45, ha="right")
plt.yticks(range(10), classes)
for i in range(10):
    for j in range(10):
        plt.text(j, i, mc[i, j], ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig("results/cifar10/confusion_matrix.png")
plt.close()

print("Modèle, courbes et matrice de confusion sauvegardés.")

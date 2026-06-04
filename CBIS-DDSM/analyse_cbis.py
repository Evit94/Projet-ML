#ANALYSE DU MODELE CBIS-DDSM
#On relit les résultats sauvegardés par CNN_cbis.py (metrics.json) et on produit
#la matrice de confusion, la courbe ROC et les courbes d'entraînement.
#On ajoute une analyse médicale (faux positifs / faux négatifs).

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

#Chargement des résultats
with open("results/cbis_ddsm/metrics.json") as f:
    data = json.load(f)

metrics = data["metrics"]
train_loss = data["train_loss"]
val_loss = data["val_loss"]
y_true = np.array(data["y_true"])
y_proba = np.array(data["y_proba"])
y_pred = (y_proba >= 0.5).astype(int)

print("=== Métriques (test) ===")
for nom, val in metrics.items():
    print(f"{nom:<10} : {val}")


#---------- MATRICE DE CONFUSION ----------
mc = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = mc.ravel()

plt.figure(figsize=(6, 5))
plt.imshow(mc, cmap="Blues")
plt.colorbar()
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.title("Matrice de confusion - CBIS-DDSM")
plt.xticks([0, 1], ["bénin", "malin"])
plt.yticks([0, 1], ["bénin", "malin"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, mc[i, j], ha="center", va="center", fontsize=12)
plt.tight_layout()
plt.savefig("results/cbis_ddsm/confusion_matrix.png")
plt.close()


#---------- COURBE ROC ----------
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']}")
plt.plot([0, 1], [0, 1], "--", color="gray")  #classifieur aléatoire
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - CBIS-DDSM")
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()
plt.savefig("results/cbis_ddsm/roc_curve.png")
plt.close()


#---------- COURBES D'ENTRAINEMENT ----------
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, "o-", label="train")
plt.plot(range(1, len(val_loss) + 1), val_loss, "o-", label="validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CBIS-DDSM - Courbes d'entraînement")
plt.legend()
plt.grid(True, ls=":")
plt.tight_layout()
plt.savefig("results/cbis_ddsm/training_curves.png")
plt.close()

print("\nGraphiques sauvegardés dans results/cbis_ddsm/")


#---------- ANALYSE MEDICALE ----------
print("\n=== Analyse médicale ===")
print(f"Vrais négatifs (bénin bien classé)  : {tn}")
print(f"Faux positifs (bénin -> malin)      : {fp}")
print(f"Faux négatifs (malin -> bénin)      : {fn}")
print(f"Vrais positifs (malin bien classé)  : {tp}")
print()
print("Les FAUX NEGATIFS (cancers manqués) sont les plus graves cliniquement :")
print("un cas malin classé bénin retarde le diagnostic et la prise en charge.")
print("Les FAUX POSITIFS entraînent des examens complémentaires inutiles et du")
print("stress, mais sont moins dangereux. En contexte médical on privilégie donc")
print("un fort RECALL (sensibilité) quitte à accepter plus de faux positifs.")

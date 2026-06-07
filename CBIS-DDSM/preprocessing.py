#PRETRAITEMENT CBIS-DDSM (mammographies)
#Classification binaire : MALIGNANT (1) contre BENIGN / BENIGN_WITHOUT_CALLBACK (0)
#
#REMARQUE IMPORTANTE :
#Le dataset CBIS-DDSM complet (images DICOM) pèse plus de 150 Go et n'est pas
#présent dans cet environnement. Ce script lit le fichier
#   mass_case_description_train_set.csv
#et charge les images aux chemins indiqués (colonne "image file path").
#Si le CSV ou les images sont absents, on génère un jeu de données SYNTHETIQUE
#(clairement signalé) afin que toute la chaîne reste exécutable et produise
#des résultats. Pour utiliser les vraies données, il suffit de placer le CSV
#et le dossier d'images à la racine du projet.

import os
import numpy as np

TAILLE = 128  #redimensionnement par défaut 128x128
CSV_PATH = "CBIS-DDSM/mass_case_description_train_set.csv"
IMG_DIR = "CBIS-DDSM/images"  #dossier racine des images réelles


def label_binaire(pathology):
    #BENIGN et BENIGN_WITHOUT_CALLBACK -> 0, MALIGNANT -> 1
    if str(pathology).strip().upper() == "MALIGNANT":
        return 1
    return 0


def charger_donnees_reelles():
    #Chargement à partir du CSV et des images réelles (si disponibles)
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(CSV_PATH)
    #La colonne de label peut s'appeler "pathology"
    col_label = "pathology"
    col_image = "image file path"

    X = []
    y = []
    ignores = 0
    for _, ligne in df.iterrows():
        chemin = os.path.join(IMG_DIR, str(ligne[col_image]).strip())
        #Nettoyage des entrées invalides
        if not os.path.exists(chemin):
            ignores += 1
            continue
        try:
            img = Image.open(chemin).convert("L").resize((TAILLE, TAILLE))
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(label_binaire(ligne[col_label]))
        except Exception:
            ignores += 1
            continue

    print(f"Images chargées : {len(X)}, entrées ignorées : {ignores}")
    X = np.array(X).reshape(-1, 1, TAILLE, TAILLE)
    y = np.array(y, dtype=np.float32)
    return X, y


def ajouter_tache(img, contraste):
    #Ajoute une tache claire (masse) à une position et taille aléatoires
    cx, cy = np.random.randint(30, 98, size=2)
    r = np.random.randint(6, 16)
    yy, xx = np.ogrid[:TAILLE, :TAILLE]
    masque = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[masque] += contraste
    return img


def generer_donnees_synthetiques(n=1500):
    #Données synthétiques de secours (le vrai dataset n'est pas disponible).
    #Pour rester réaliste, les deux classes peuvent contenir une tache, mais avec
    #des contrastes qui se chevauchent : les masses malignes sont en moyenne plus
    #marquées que les masses bénignes. Avec un fond bruité, la séparation n'est pas
    #parfaite, ce qui donne des faux positifs / faux négatifs exploitables.
    print("ATTENTION : CSV/images CBIS-DDSM introuvables.")
    print("-> Génération d'un jeu de données SYNTHETIQUE pour la démonstration.")
    np.random.seed(42)
    #Déséquilibre volontaire (~30% malins) pour reproduire le cas réel
    n_malin = int(n * 0.3)
    n_benin = n - n_malin

    X = []
    y = []
    #Images bénignes : fond bruité, parfois une masse de faible contraste
    for _ in range(n_benin):
        img = np.random.normal(0.5, 0.10, (TAILLE, TAILLE)).astype(np.float32)
        if np.random.rand() < 0.4:
            img = ajouter_tache(img, np.random.uniform(0.04, 0.14))
        X.append(np.clip(img, 0, 1))
        y.append(0)
    #Images malignes : fond bruité + masse de contraste plus élevé (mais variable)
    for _ in range(n_malin):
        img = np.random.normal(0.5, 0.10, (TAILLE, TAILLE)).astype(np.float32)
        img = ajouter_tache(img, np.random.uniform(0.14, 0.32))
        X.append(np.clip(img, 0, 1))
        y.append(1)

    X = np.array(X).reshape(-1, 1, TAILLE, TAILLE)
    y = np.array(y, dtype=np.float32)
    #Mélange
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


#Chargement (réel si possible, sinon synthétique)
if os.path.exists(CSV_PATH) and os.path.isdir(IMG_DIR):
    X, y = charger_donnees_reelles()
else:
    X, y = generer_donnees_synthetiques()


#Split train / validation / test (70 / 15 / 15)
np.random.seed(0)
n = len(X)
idx = np.random.permutation(n)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

idx_train = idx[:n_train]
idx_val = idx[n_train:n_train + n_val]
idx_test = idx[n_train + n_val:]

X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
X_test, y_test = X[idx_test], y[idx_test]


#Poids des classes pour gérer le déséquilibre (utilisé par BCEWithLogitsLoss)
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
pos_weight = n_neg / max(n_pos, 1)  #poids appliqué à la classe positive (maligne)


def afficher_repartition():
    print("\n=== Répartition des classes ===")
    for nom, yy in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        n_b = int(np.sum(yy == 0))
        n_m = int(np.sum(yy == 1))
        print(f"{nom:<12} : bénin = {n_b}, malin = {n_m} (total {len(yy)})")
    print(f"pos_weight (déséquilibre) = {pos_weight:.2f}")


if __name__ == "__main__":
    print(f"Taille des images : {TAILLE}x{TAILLE}")
    print(f"Total : {n} images")
    afficher_repartition()

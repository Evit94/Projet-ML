#PRETRAITEMENT CBIS-DDSM (mammographies)
#Classification binaire : MALIGNANT (1) contre BENIGN / BENIGN_WITHOUT_CALLBACK (0)
#
#UTILISATION AVEC LES VRAIES DONNEES :
#1. Placer le dossier contenant les images DICOM dans CBIS-DDSM/images/
#   (les chemins dans le CSV sont du type Mass-Training_P_00001_LEFT_CC/.../000000.dcm)
#2. Le CSV mass_case_description_train_set.csv est déjà dans CBIS-DDSM/
#3. Lancer depuis la racine du projet : python CBIS-DDSM/preprocessing.py
#
#Si les images ne sont pas trouvées, le script bascule sur des données synthétiques.

import os
import numpy as np

TAILLE = 128  #redimensionnement par défaut 128x128

#Chemin du CSV (relatif à la racine du projet)
CSV_PATH = "CBIS-DDSM/mass_case_description_train_set.csv"
#Dossier racine des images DICOM (les chemins du CSV s'y appliquent directement)
IMG_DIR = "CBIS-DDSM/images"


def label_binaire(pathology):
    #BENIGN et BENIGN_WITHOUT_CALLBACK -> 0, MALIGNANT -> 1
    if str(pathology).strip().upper() == "MALIGNANT":
        return 1
    return 0


def lire_dicom(chemin):
    #Lecture d'un fichier DICOM et conversion en tableau numpy normalisé [0,1]
    import pydicom
    from PIL import Image

    ds = pydicom.dcmread(chemin)
    pixel = ds.pixel_array.astype(np.float32)
    #Normalisation min-max sur l'image (mammographies: dynamique très variable)
    pmin, pmax = pixel.min(), pixel.max()
    if pmax > pmin:
        pixel = (pixel - pmin) / (pmax - pmin)
    #Redimensionnement
    img = Image.fromarray((pixel * 255).astype(np.uint8)).convert("L")
    img = img.resize((TAILLE, TAILLE))
    return np.asarray(img, dtype=np.float32) / 255.0


def charger_donnees_reelles():
    import pandas as pd

    df = pd.read_csv(CSV_PATH)

    #On utilise le recadrage sur la masse (cropped image file path)
    #plutôt que la mammographie complète : plus pertinent pour la classification
    col_label = "pathology"
    col_image = "cropped image file path"

    X = []
    y = []
    ignores = 0

    for _, ligne in df.iterrows():
        chemin_relatif = str(ligne[col_image]).strip()
        chemin = os.path.join(IMG_DIR, chemin_relatif)

        if not os.path.exists(chemin):
            ignores += 1
            continue
        try:
            img = lire_dicom(chemin)
            X.append(img)
            y.append(label_binaire(ligne[col_label]))
        except Exception as e:
            ignores += 1
            continue

    if len(X) == 0:
        return None, None

    print(f"Images chargées : {len(X)}, entrées ignorées (chemin manquant) : {ignores}")
    X = np.array(X).reshape(-1, 1, TAILLE, TAILLE)
    y = np.array(y, dtype=np.float32)
    return X, y


def ajouter_tache(img, contraste):
    cx, cy = np.random.randint(30, 98, size=2)
    r = np.random.randint(6, 16)
    yy, xx = np.ogrid[:TAILLE, :TAILLE]
    masque = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[masque] += contraste
    return img


def generer_donnees_synthetiques(n=1500):
    #Données synthétiques de secours (images DICOM non trouvées).
    #Les masses malignes ont un contraste légèrement plus élevé que les bénignes,
    #avec chevauchement volontaire pour générer de vrais faux positifs / faux négatifs.
    print("Images DICOM non trouvées dans", IMG_DIR)
    print("-> Génération d'un jeu de données SYNTHETIQUE pour la démonstration.")
    print("   Pour les vraies données : placer les dossiers DICOM dans CBIS-DDSM/images/")
    np.random.seed(42)

    n_malin = int(n * 0.3)
    n_benin = n - n_malin

    X, y = [], []
    for _ in range(n_benin):
        img = np.random.normal(0.5, 0.18, (TAILLE, TAILLE)).astype(np.float32)
        if np.random.rand() < 0.5:
            img = ajouter_tache(img, np.random.uniform(0.05, 0.18))
        X.append(np.clip(img, 0, 1))
        y.append(0)
    for _ in range(n_malin):
        img = np.random.normal(0.5, 0.18, (TAILLE, TAILLE)).astype(np.float32)
        img = ajouter_tache(img, np.random.uniform(0.12, 0.30))
        X.append(np.clip(img, 0, 1))
        y.append(1)

    X = np.array(X).reshape(-1, 1, TAILLE, TAILLE)
    y = np.array(y, dtype=np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


#---------- Chargement ----------
#On cherche d'abord les vraies images DICOM
_donnees_reelles = False
if os.path.exists(CSV_PATH) and os.path.isdir(IMG_DIR):
    print("CSV trouvé. Chargement des images DICOM...")
    X, y = charger_donnees_reelles()
    if X is not None and len(X) > 0:
        _donnees_reelles = True
    else:
        print("Aucune image DICOM valide trouvée.")

if not _donnees_reelles:
    X, y = generer_donnees_synthetiques()


#---------- Split train / validation / test (70 / 15 / 15) ----------
np.random.seed(0)
n = len(X)
idx = np.random.permutation(n)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

idx_train = idx[:n_train]
idx_val   = idx[n_train:n_train + n_val]
idx_test  = idx[n_train + n_val:]

X_train, y_train = X[idx_train], y[idx_train]
X_val,   y_val   = X[idx_val],   y[idx_val]
X_test,  y_test  = X[idx_test],  y[idx_test]

#Poids pour BCEWithLogitsLoss (gestion du déséquilibre)
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
pos_weight = n_neg / max(n_pos, 1)


def afficher_repartition():
    print("\n=== Répartition des classes ===")
    for nom, yy in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        n_b = int(np.sum(yy == 0))
        n_m = int(np.sum(yy == 1))
        print(f"{nom:<12} : bénin = {n_b}, malin = {n_m} (total {len(yy)})")
    print(f"pos_weight (déséquilibre) = {pos_weight:.2f}")


if __name__ == "__main__":
    print(f"Taille des images : {TAILLE}x{TAILLE}")
    print(f"Total chargé : {n} images")
    afficher_repartition()

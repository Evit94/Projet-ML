#PRETRAITEMENT CBIS-DDSM (mammographies)
#Classification binaire : MALIGNANT (1) contre BENIGN / BENIGN_WITHOUT_CALLBACK (0)
#
#Charge les 4 CSV (mass + calc, train + test) et utilise dicom_info.csv pour
#retrouver le chemin JPEG correspondant à chaque entrée (mammographies complètes).
#Les images sont dans archive/jpeg/{series_uid}/{fichier}.jpg

import os
import numpy as np
import pandas as pd
from PIL import Image

TAILLE = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR  = os.path.join(BASE_DIR, "archive", "csv")
JPEG_DIR = os.path.join(BASE_DIR, "archive", "jpeg")


def label_binaire(pathology):
    #BENIGN et BENIGN_WITHOUT_CALLBACK -> 0, MALIGNANT -> 1
    if str(pathology).strip().upper() == "MALIGNANT":
        return 1
    return 0


def construire_mapping_jpeg():
    #Construit un dict PatientID -> chemin JPEG absolu (mammographies complètes uniquement)
    dicom_info = pd.read_csv(os.path.join(CSV_DIR, "dicom_info.csv"))
    full_mam = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"]
    mapping = {}
    for _, row in full_mam.iterrows():
        patient_id  = str(row["PatientID"]).strip()
        img_path_raw = str(row["image_path"]).strip()
        #image_path format dans dicom_info : "CBIS-DDSM/jpeg/{series_uid}/{fichier}.jpg"
        #On extrait la partie après "CBIS-DDSM/jpeg/" et on joint avec JPEG_DIR
        relative = img_path_raw.replace("CBIS-DDSM/jpeg/", "").lstrip("/")
        full_path = os.path.join(JPEG_DIR, relative.replace("/", os.sep))
        mapping[patient_id] = full_path
    return mapping


def charger_donnees_reelles():
    mapping = construire_mapping_jpeg()

    csv_files = [
        os.path.join(CSV_DIR, "mass_case_description_train_set.csv"),
        os.path.join(CSV_DIR, "mass_case_description_test_set.csv"),
        os.path.join(CSV_DIR, "calc_case_description_train_set.csv"),
        os.path.join(CSV_DIR, "calc_case_description_test_set.csv"),
    ]

    X = []
    y = []
    ignores  = 0
    seen_paths = set()  #évite de charger la même image plusieurs fois (plusieurs abnormalities par image)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, ligne in df.iterrows():
            #Le PatientID correspond au premier composant du chemin "image file path"
            img_file_path = str(ligne["image file path"]).strip().replace("\n", "").replace("\r", "")
            patient_id    = img_file_path.split("/")[0].strip()

            if patient_id not in mapping:
                ignores += 1
                continue

            chemin = mapping[patient_id]

            if chemin in seen_paths:
                continue
            seen_paths.add(chemin)

            if not os.path.exists(chemin):
                ignores += 1
                continue

            try:
                img = Image.open(chemin).convert("L").resize((TAILLE, TAILLE))
                X.append(np.asarray(img, dtype=np.float32) / 255.0)
                y.append(label_binaire(ligne["pathology"]))
            except Exception:
                ignores += 1
                continue

    print(f"Images chargées : {len(X)}, entrées ignorées : {ignores}")
    X = np.array(X).reshape(-1, 1, TAILLE, TAILLE)
    y = np.array(y, dtype=np.float32)
    return X, y


#Chargement
X, y = charger_donnees_reelles()

#Split train / validation / test (70 / 15 / 15)
np.random.seed(0)
n   = len(X)
idx = np.random.permutation(n)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

idx_train = idx[:n_train]
idx_val   = idx[n_train:n_train + n_val]
idx_test  = idx[n_train + n_val:]

X_train, y_train = X[idx_train], y[idx_train]
X_val,   y_val   = X[idx_val],   y[idx_val]
X_test,  y_test  = X[idx_test],  y[idx_test]

#Poids des classes pour gérer le déséquilibre (utilisé par BCEWithLogitsLoss)
n_pos      = np.sum(y_train == 1)
n_neg      = np.sum(y_train == 0)
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
    print(f"Total : {n} images")
    afficher_repartition()

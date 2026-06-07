#PRETRAITEMENT CBIS-DDSM (mammographies)
#Classification binaire : MALIGNANT (1) contre BENIGN / BENIGN_WITHOUT_CALLBACK (0)
#
#On respecte le SPLIT OFFICIEL CBIS-DDSM :
#  - *_train_set.csv -> entraînement (+ validation découpée PAR PATIENT)
#  - *_test_set.csv  -> test
#Ce découpage par patient évite la fuite de données : un même patient possède
#plusieurs vues quasi identiques (LEFT_CC, LEFT_MLO, RIGHT_CC, RIGHT_MLO) et ne
#doit jamais se retrouver à la fois dans le train et dans le test/validation.
#
#dicom_info.csv sert à retrouver le chemin JPEG de chaque vue (mammographie complète).
#Les images sont dans archive/jpeg/{series_uid}/{fichier}.jpg

import os
import re
import numpy as np
import pandas as pd
from PIL import Image

TAILLE = 128
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_DIR   = os.path.join(BASE_DIR, "archive", "csv")
JPEG_DIR  = os.path.join(BASE_DIR, "archive", "jpeg")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

#CSV du split officiel
TRAIN_CSV = [
    os.path.join(CSV_DIR, "mass_case_description_train_set.csv"),
    os.path.join(CSV_DIR, "calc_case_description_train_set.csv"),
]
TEST_CSV = [
    os.path.join(CSV_DIR, "mass_case_description_test_set.csv"),
    os.path.join(CSV_DIR, "calc_case_description_test_set.csv"),
]


def label_binaire(pathology):
    #BENIGN et BENIGN_WITHOUT_CALLBACK -> 0, MALIGNANT -> 1
    if str(pathology).strip().upper() == "MALIGNANT":
        return 1
    return 0


def patient_de_base(view_id):
    #"Mass-Training_P_00001_LEFT_CC" -> "P_00001"
    #Permet de regrouper toutes les vues d'un même patient.
    m = re.search(r"(P_\d+)", view_id)
    return m.group(1) if m else view_id


def construire_mapping_jpeg():
    #Construit un dict identifiant_vue -> chemin JPEG absolu (mammographies complètes uniquement)
    dicom_info = pd.read_csv(os.path.join(CSV_DIR, "dicom_info.csv"))
    full_mam = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"]
    mapping = {}
    for _, row in full_mam.iterrows():
        view_id      = str(row["PatientID"]).strip()
        img_path_raw = str(row["image_path"]).strip()
        #image_path format dans dicom_info : "CBIS-DDSM/jpeg/{series_uid}/{fichier}.jpg"
        relative = img_path_raw.replace("CBIS-DDSM/jpeg/", "").lstrip("/")
        full_path = os.path.join(JPEG_DIR, relative.replace("/", os.sep))
        mapping[view_id] = full_path
    return mapping


def charger_set(csv_files, mapping):
    #Charge les images d'une liste de CSV. Retourne X, y et les patients (pour le groupage).
    X, y, groupes = [], [], []
    ignores = 0
    seen_paths = set()  #évite de charger 2x la même vue (plusieurs anomalies par image)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, ligne in df.iterrows():
            #L'identifiant de vue est le premier composant du chemin "image file path"
            img_file_path = str(ligne["image file path"]).strip().replace("\n", "").replace("\r", "")
            view_id       = img_file_path.split("/")[0].strip()

            if view_id not in mapping:
                ignores += 1
                continue

            chemin = mapping[view_id]
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
                groupes.append(patient_de_base(view_id))
                if len(X) % 50 == 0:
                    print(f"  ... {len(X)} images chargées", flush=True)
            except Exception:
                ignores += 1
                continue

    print(f"  -> {len(X)} images chargées, {ignores} entrées ignorées")
    X = np.array(X).reshape(-1, 1, TAILLE, TAILLE)
    y = np.array(y, dtype=np.float32)
    groupes = np.array(groupes)
    return X, y, groupes


def decouper_train_val(X, y, groupes, val_frac=0.15, seed=0):
    #Découpe le train officiel en train/validation PAR PATIENT (aucun patient partagé).
    rng = np.random.RandomState(seed)
    patients = np.unique(groupes)
    rng.shuffle(patients)
    n_val = int(val_frac * len(patients))
    patients_val = set(patients[:n_val])

    masque_val = np.array([g in patients_val for g in groupes])
    X_tr, y_tr = X[~masque_val], y[~masque_val]
    X_vl, y_vl = X[masque_val],  y[masque_val]
    return X_tr, y_tr, X_vl, y_vl


#Chargement avec cache NPY (par ensemble du split officiel) pour éviter de tout reprocesser
os.makedirs(CACHE_DIR, exist_ok=True)
_cache = {
    "Xtr": os.path.join(CACHE_DIR, f"X_train_off_{TAILLE}.npy"),
    "ytr": os.path.join(CACHE_DIR, f"y_train_off_{TAILLE}.npy"),
    "gtr": os.path.join(CACHE_DIR, f"g_train_off_{TAILLE}.npy"),
    "Xte": os.path.join(CACHE_DIR, f"X_test_off_{TAILLE}.npy"),
    "yte": os.path.join(CACHE_DIR, f"y_test_off_{TAILLE}.npy"),
}

if all(os.path.exists(p) for p in _cache.values()):
    print("Cache trouvé, chargement depuis le disque...")
    X_train_full = np.load(_cache["Xtr"])
    y_train_full = np.load(_cache["ytr"])
    g_train_full = np.load(_cache["gtr"])
    X_test       = np.load(_cache["Xte"])
    y_test       = np.load(_cache["yte"])
    print(f"Train officiel : {len(X_train_full)} images, Test officiel : {len(X_test)} images")
else:
    mapping = construire_mapping_jpeg()
    print("Chargement du TRAIN officiel (mass + calc train)...")
    X_train_full, y_train_full, g_train_full = charger_set(TRAIN_CSV, mapping)
    print("Chargement du TEST officiel (mass + calc test)...")
    X_test, y_test, _ = charger_set(TEST_CSV, mapping)
    np.save(_cache["Xtr"], X_train_full)
    np.save(_cache["ytr"], y_train_full)
    np.save(_cache["gtr"], g_train_full)
    np.save(_cache["Xte"], X_test)
    np.save(_cache["yte"], y_test)
    print("Cache NPY sauvegardé.")

#Validation découpée PAR PATIENT depuis le train officiel
X_train, y_train, X_val, y_val = decouper_train_val(
    X_train_full, y_train_full, g_train_full, val_frac=0.15, seed=0
)

#Poids des classes pour gérer le déséquilibre (utilisé par BCEWithLogitsLoss)
n_pos      = np.sum(y_train == 1)
n_neg      = np.sum(y_train == 0)
pos_weight = n_neg / max(n_pos, 1)


def afficher_repartition():
    print("\n=== Répartition des classes (split officiel) ===")
    for nom, yy in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        n_b = int(np.sum(yy == 0))
        n_m = int(np.sum(yy == 1))
        print(f"{nom:<12} : bénin = {n_b}, malin = {n_m} (total {len(yy)})")
    print(f"pos_weight (déséquilibre) = {pos_weight:.2f}")


if __name__ == "__main__":
    print(f"Taille des images : {TAILLE}x{TAILLE}")
    print(f"Train+val : {len(X_train_full)}, Test : {len(X_test)}")
    afficher_repartition()

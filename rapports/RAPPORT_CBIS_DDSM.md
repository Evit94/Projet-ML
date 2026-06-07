# Rapport CBIS-DDSM — Classification bénin / malin

## Objectif

Classification binaire de mammographies (masses) : distinguer les cas
**MALIGNANT** (malin, label 1) des cas **BENIGN** et **BENIGN_WITHOUT_CALLBACK**
(bénin, label 0).

## Données utilisées

On utilise les **vraies données CBIS-DDSM** (version JPEG du dataset), présentes dans
`CBIS-DDSM/archive/` (CSV de description des cas + `dicom_info.csv` + images JPEG).
Le script `preprocessing.py` :

- lit les CSV de description (mass + calc), construit le label binaire à partir de la
  colonne *pathology* (MALIGNANT = 1, sinon 0) ;
- retrouve le chemin JPEG de chaque vue (mammographie complète) via `dicom_info.csv`
  (`SeriesDescription == "full mammogram images"`) ;
- redimensionne en 128×128 (niveaux de gris) et normalise les pixels dans [0, 1] ;
- déduplique les vues (une même image peut porter plusieurs anomalies).

Au total **2 857 vues** (mammographies complètes) sont chargées, issues de
**1 460 patients distincts** — chaque patient possédant plusieurs vues (LEFT/RIGHT ×
CC/MLO). Ce détail est crucial pour le découpage (voir ci-dessous).

## Attention à la fuite de données (data leakage)

Une première version découpait train/test **aléatoirement par image**. Résultat :
AUC ≈ 0.995, accuracy ≈ 0.96 — beaucoup trop beau. La cause était une **fuite de
données** : les vues quasi identiques d'un même patient (LEFT_CC, LEFT_MLO…) se
retrouvaient à la fois dans le train et dans le test. Le modèle reconnaissait des
patients déjà vus au lieu de généraliser.

**Correction** : on respecte désormais le **split officiel CBIS-DDSM** (les CSV
`*_train_set` servent à l'entraînement, les `*_test_set` au test), et la validation
est découpée **par patient** depuis le train (aucun patient partagé entre train,
validation et test). Les métriques chutent alors à des valeurs réalistes (voir plus
bas) — c'est le prix de l'honnêteté méthodologique.

## Prétraitement

- Association image ↔ label (binaire : MALIGNANT = 1, sinon 0)
- Nettoyage des entrées invalides (chemins manquants ou images illisibles ignorés)
- Redimensionnement configurable, 128×128 par défaut
- Normalisation des pixels dans [0, 1]
- **Split officiel** : train officiel = entraînement, test officiel = test
- **Validation découpée par patient** depuis le train (15% des patients), pour éviter
  toute fuite entre train et validation

Répartition obtenue (données réelles, split officiel) :

| Ensemble | bénin | malin | total |
|----------|-------|-------|-------|
| train | 1165 | 921 | 2086 |
| validation | 189 | 183 | 372 |
| test | 235 | 164 | 399 |

## Gestion du déséquilibre

Les classes sont légèrement déséquilibrées (~44% de malins dans le train). On utilise
une **Weighted Cross Entropy** : `BCEWithLogitsLoss(pos_weight = n_negatifs / n_positifs)`
(≈ 1.26 sur le train officiel). La classe minoritaire (maligne) pèse donc un peu plus
dans la fonction de coût. La répartition des classes est affichée au lancement
(`afficher_repartition`).

## Architecture du modèle

CNN simple (PyTorch), dans le style du CNN CIFAR-10 :

```
Conv2d(1, 32, 3) -> ReLU -> MaxPool(2)      # 128 -> 64
Conv2d(32, 64, 3) -> ReLU -> MaxPool(2)     # 64 -> 32
Conv2d(64, 64, 3) -> ReLU -> MaxPool(2)     # 32 -> 16
Flatten -> Linear(64*16*16, 128) -> ReLU -> Linear(128, 1)  # logit
```

- Activation : ReLU
- Optimiseur : Adam (lr = 0.001)
- Fonction de coût : BCEWithLogitsLoss avec `pos_weight`
- **Early stopping** sur la loss de validation (patience = 5)
- **Sauvegarde du meilleur modèle** : `CBIS-DDSM/params/cnn_cbis_best.pth`

## Évaluation

Métriques sur le test (`results/cbis_ddsm/metrics.json`) :

| Métrique | Valeur |
|----------|--------|
| Accuracy | 0.602 |
| Precision | 0.512 |
| Recall (sensibilité) | 0.634 |
| F1-score | 0.567 |
| ROC-AUC | 0.658 |

Ces valeurs, bien plus modestes que la version « avec fuite » (AUC 0.995), sont
**réalistes** pour un petit CNN sur mammographies complètes 128×128 sans transfer
learning ni augmentation de données. La littérature atteint ~0.7–0.85 d'AUC, mais
avec des architectures pré-entraînées, une résolution plus élevée et des patchs de
lésions (ROI).

Graphiques (`results/cbis_ddsm/`) :
- `confusion_matrix.png`
- `roc_curve.png`
- `training_curves.png` (loss train vs validation)

Matrice de confusion (test) :

|              | Prédit bénin | Prédit malin |
|--------------|--------------|--------------|
| **Vrai bénin** | 136 (VN) | 99 (FP) |
| **Vrai malin** | 60 (FN) | 104 (VP) |

## Analyse médicale

**Faux positifs (99)** : des cas bénins classés malins (~42% des bénins). Conséquence :
examens complémentaires inutiles (biopsie, imagerie supplémentaire), anxiété pour la
patiente, coût. C'est gênant mais **pas dangereux**.

**Faux négatifs (60)** : des cas malins classés bénins, soit ~37% des cancers du test
non détectés. C'est l'erreur **la plus grave** : un cancer manqué retarde le
diagnostic et la prise en charge, ce qui peut mettre en jeu le pronostic vital. Ce
taux élevé montre que le modèle, en l'état, est **loin d'être utilisable en clinique**.

**Compromis clinique.** En dépistage, on cherche à **minimiser les faux négatifs**,
quitte à accepter davantage de faux positifs. On privilégie donc un fort **recall
(sensibilité)**. La pondération de la classe maligne (`pos_weight`) va dans ce sens.
On pourrait aussi abaisser le seuil de décision (< 0.5) pour augmenter encore la
sensibilité, au prix de plus de faux positifs — la courbe ROC permet de choisir ce
compromis.

## Limites du modèle

- **CNN simple** sans transfer learning ni augmentation de données : c'est la
  principale limite de performance. Des architectures pré-entraînées (ResNet,
  EfficientNet) atteindraient des AUC nettement supérieures.
- Résolution réduite à 128×128 : on perd des micro-détails (micro-calcifications)
  pourtant cliniquement importants.
- Mammographies **complètes** (et non des patchs de lésions / ROI) : la lésion ne
  représente qu'une petite portion de l'image, ce qui complique l'apprentissage.
- Un seul split : pas de validation croisée, donc estimation de performance peu robuste.
- Certaines vues de test n'ont pas pu être appariées via `dicom_info.csv` (identifiants
  avec suffixe) et sont ignorées — le test réel pourrait être un peu plus grand.

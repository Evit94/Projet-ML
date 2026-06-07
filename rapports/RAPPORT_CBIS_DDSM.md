# Rapport CBIS-DDSM — Classification bénin / malin

## Objectif

Classification binaire de mammographies (masses) : distinguer les cas
**MALIGNANT** (malin, label 1) des cas **BENIGN** et **BENIGN_WITHOUT_CALLBACK**
(bénin, label 0).

## Remarque sur les données

Le dataset CBIS-DDSM complet (images DICOM) pèse plus de 150 Go et n'est pas
disponible dans l'environnement d'exécution. Le script `preprocessing.py` est
écrit pour les **vraies données** : il lit `mass_case_description_train_set.csv`,
associe chaque ligne à son image (colonne *image file path*), construit le label
binaire à partir de la colonne *pathology*, redimensionne en 128×128 et normalise.

En l'absence du CSV et des images, le script bascule automatiquement sur un jeu de
données **synthétique** (clairement signalé à l'exécution) afin que toute la chaîne
reste exécutable et produise des résultats. Les images synthétiques reproduisent les
caractéristiques utiles à la démonstration : fond bruité, masses bénignes de faible
contraste, masses malignes de contraste plus élevé mais variable (chevauchement
volontaire pour générer de vrais faux positifs / faux négatifs), et un déséquilibre
des classes réaliste (~30% de malins).

## Prétraitement

- Association image ↔ label (binaire : MALIGNANT = 1, sinon 0)
- Nettoyage des entrées invalides (chemins manquants ou images illisibles ignorés)
- Redimensionnement configurable, 128×128 par défaut
- Normalisation des pixels dans [0, 1]
- Découpage train / validation / test (70 / 15 / 15)

Répartition obtenue (exemple synthétique) :

| Ensemble | bénin | malin | total |
|----------|-------|-------|-------|
| train | 745 | 305 | 1050 |
| validation | ~160 | ~65 | 225 |
| test | 160 | 65 | 225 |

## Gestion du déséquilibre

Les classes étant déséquilibrées (~30% de malins), on utilise une **Weighted Cross
Entropy** : `BCEWithLogitsLoss(pos_weight = n_negatifs / n_positifs)` (≈ 2.25). La
classe minoritaire (maligne) pèse donc davantage dans la fonction de coût. La
répartition des classes est affichée au lancement (`afficher_repartition`).

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
| Accuracy | 0.956 |
| Precision | 0.923 |
| Recall (sensibilité) | 0.923 |
| F1-score | 0.923 |
| ROC-AUC | 0.995 |

Graphiques (`results/cbis_ddsm/`) :
- `confusion_matrix.png`
- `roc_curve.png`
- `training_curves.png` (loss train vs validation)

Matrice de confusion (test) :

|              | Prédit bénin | Prédit malin |
|--------------|--------------|--------------|
| **Vrai bénin** | 155 (VN) | 5 (FP) |
| **Vrai malin** | 5 (FN) | 60 (VP) |

## Analyse médicale

**Faux positifs (5)** : des cas bénins classés malins. Conséquence : examens
complémentaires inutiles (biopsie, imagerie supplémentaire), anxiété pour la
patiente, coût. C'est gênant mais **pas dangereux**.

**Faux négatifs (5)** : des cas malins classés bénins, soit ~8% des cancers du test
non détectés. C'est l'erreur **la plus grave** : un cancer manqué retarde le
diagnostic et la prise en charge, ce qui peut mettre en jeu le pronostic vital.

**Compromis clinique.** En dépistage, on cherche à **minimiser les faux négatifs**,
quitte à accepter davantage de faux positifs. On privilégie donc un fort **recall
(sensibilité)**. La pondération de la classe maligne (`pos_weight`) va dans ce sens.
On pourrait aussi abaisser le seuil de décision (< 0.5) pour augmenter encore la
sensibilité, au prix de plus de faux positifs — la courbe ROC permet de choisir ce
compromis.

## Limites du modèle

- **Données synthétiques** : les performances réelles sur CBIS-DDSM seraient bien plus
  faibles (les vraies mammographies sont beaucoup plus difficiles, AUC typiques
  ~0.7–0.85 dans la littérature).
- Jeu de données réduit et CNN simple (pas de transfer learning, pas d'augmentation
  de données).
- Résolution réduite à 128×128 : on perd des micro-détails (micro-calcifications)
  pourtant cliniquement importants.
- Un seul split : pas de validation croisée, donc estimation de performance peu robuste.

# Rapport CIFAR-10 — Comparaison des modèles

## Objectif

Classifier les images CIFAR-10 (10 classes : avion, auto, oiseau, chat, cerf,
chien, grenouille, cheval, bateau, camion) et comparer plusieurs approches :
modèles linéaires, perceptrons à une couche cachée (MLP) et réseau convolutif (CNN).
On compare aussi l'effet du passage en niveaux de gris (1024 entrées) vs couleur
(3072 entrées).

## Scripts

- `preprocessing.py` : chargement CIFAR-10, conversion en niveaux de gris
  (0.299R + 0.587G + 0.114B), reshape, normalisation, one-hot encoding.
- `CIFAR_10_gris_lineaire.py`, `CIFAR_10_gris_couche_cachee.py`,
  `CIFAR_10_couleur_lineaire.py`, `CIFAR_10_couleur_couche_cachee.py` : les 4 modèles
  NumPy déjà présents (linéaire / 1 couche, gris / couleur).
- `CNN.py` : CNN PyTorch (Conv ×4 + MaxPool ×2 + Dense), avec **early stopping** et
  **sauvegarde du meilleur modèle** (`params/cnn_best.pth`). Produit les courbes de
  loss et d'accuracy ainsi que la matrice de confusion.
- `analyse_cifar.py` : réentraîne brièvement les 4 modèles NumPy (sous-ensemble de
  20 000 images), récupère les métriques du CNN, et produit la comparaison globale.

## Résultats

Métriques d'accuracy (`results/cifar10/metrics.json`). Les modèles NumPy sont
entraînés sur un sous-ensemble (20 000 images) pour limiter le temps de calcul ;
les tendances restent représentatives.

| Modèle | Accuracy train | Accuracy test |
|--------|---------------:|--------------:|
| Linéaire (gris) | 20.3% | 17.7% |
| 1 couche (gris) | 37.6% | 34.8% |
| Linéaire (couleur) | 37.2% | 34.1% |
| 1 couche (couleur) | 49.8% | 43.9% |
| **CNN (couleur)** | **66.5%** | **48.6%** |

Graphiques (`results/cifar10/`) :
- `model_comparison.png` — comparaison des 5 modèles (train vs test)
- `loss_curve.png`, `accuracy_curve.png` — courbes d'entraînement du CNN
- `confusion_matrix.png` — matrice de confusion du CNN sur le test

## Comparaison linéaire / MLP / CNN

**Couleur vs gris.** À architecture égale, la couleur apporte un gain net : le
linéaire passe de 17.7% (gris) à 34.1% (couleur), le MLP de 34.8% à 43.9%. La couleur
contient une information discriminante forte (ciel bleu, herbe verte…) que la
conversion en niveaux de gris détruit.

**Linéaire vs MLP.** L'ajout d'une couche cachée (ReLU) améliore systématiquement les
performances (couleur : 34.1% → 43.9%). Le modèle linéaire ne peut tracer que des
frontières linéaires dans l'espace des pixels, insuffisant pour des objets aussi
variables.

**MLP vs CNN.** Le CNN obtient la meilleure accuracy test (48.6%). Surtout, il
exploite la **structure spatiale** de l'image (convolutions, invariance par
translation) là où le MLP traite chaque pixel indépendamment après mise à plat. Le
CNN est donc bien plus adapté aux images.

## Nombre de paramètres

| Modèle | Paramètres (approx.) |
|--------|---------------------:|
| Linéaire (gris) | 10 × 1024 + 10 ≈ **10 k** |
| Linéaire (couleur) | 10 × 3072 + 10 ≈ **31 k** |
| 1 couche (gris) | 128 × 1024 + 128 × 10 ≈ **132 k** |
| 1 couche (couleur) | 128 × 3072 + 128 × 10 ≈ **395 k** |
| CNN | 4 conv (64 filtres) + dense ≈ **154 k** |

Remarque intéressante : le CNN a *moins* de paramètres que le MLP couleur (le partage
de poids des convolutions est très économe) tout en étant plus performant. La
performance ne dépend pas que du nombre de paramètres mais de l'adéquation de
l'architecture aux données.

## Overfitting

Le surapprentissage est très visible sur le CNN : sur ses courbes (`loss_curve.png`,
`accuracy_curve.png`), l'accuracy train continue de monter (jusqu'à ~74% à l'epoch 15)
alors que l'accuracy test stagne autour de 48% à partir de l'epoch ~11. L'**early
stopping** sert précisément à éviter de continuer à entraîner dans cette zone : on
conserve le modèle au meilleur score de validation. L'écart train/test (~66% vs 49%
sur les métriques sauvegardées) confirme l'overfitting, attendu sans régularisation
ni augmentation de données.

## Difficulté de CIFAR-10 par rapport à MNIST

MNIST atteignait ~98% d'accuracy avec un simple MLP ; CIFAR-10 plafonne ici autour de
48% avec un CNN. Plusieurs raisons :

- **Images naturelles** vs chiffres : forte variabilité d'arrière-plan, d'éclairage,
  de pose, d'échelle et de point de vue.
- **Couleur et texture** : 3 canaux, objets composés de textures complexes, contre
  des chiffres binaires centrés.
- **Variabilité intra-classe** : un « chien » peut prendre mille apparences, alors
  qu'un « 7 » manuscrit reste un 7.
- **Confusions inter-classes** : la matrice de confusion montre des erreurs typiques
  entre classes proches (chat/chien, auto/camion, cerf/cheval).

CIFAR-10 est donc nettement plus difficile, et c'est précisément pour ce type de
données que les CNN font la différence par rapport aux modèles pleinement connectés.

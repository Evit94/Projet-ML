Bonjour, voici notre projet de ML (SM604) contenant les trois parties :
partie 1 (MNIST), partie 2 (CIFAR-10) et partie 3 (CBIS-DDSM).

Les résultats (graphiques, CSV, métriques) sont sauvegardés dans le dossier
"results/", et les rapports rédigés dans le dossier "rapports/".

============================================================
PARTIE 1 - MNIST (Classification de chiffres manuscrits)
============================================================

Tous les fichiers se trouvent dans le dossier "MNIST/".

- "preprocessing.py" : chargement du dataset MNIST, reshape (60000, 784),
  normalisation entre 0 et 1, et one-hot encoding des étiquettes.
  Ce fichier est importé par tous les autres fichiers MNIST.

- "MNIST_lineaire.py" : modèle linéaire sans couche cachée, implémenté
  entièrement à la main avec NumPy. Contient les fonctions softmax,
  cross_entropy, forward, gradient et train. Descente de gradient image
  par image. Taux d'erreur obtenu : ~8.2% (train) / ~8.8% (test).
  Sauvegarde les paramètres A et b dans "params/".

- "MNIST_couche_cachee.py" : réseau avec une couche cachée (p1=128 neurones,
  activation ReLU), rétropropagation implémentée à la main. Taux d'erreur
  obtenu : ~0.7% (train) / ~2.4% (test).
  Sauvegarde les paramètres W1, b1, W2, b2 dans "params/".

- "MNIST_2couche_cachee.py" : réseau avec deux couches cachées (p1=128,
  p2=64 neurones, activation ReLU), rétropropagation à la main.
  Taux d'erreur obtenu : ~1.3% (train) / ~2.6% (test).
  Sauvegarde les paramètres W1, b1, W2, b2, W3, b3 dans "params/".

- "analyse.py" : fichier d'analyse comparative des trois modèles. Charge
  les paramètres sauvegardés et produit :
    - Tableau comparatif des taux d'erreur sur le test set
    - Matrice de confusion (modèle 1 couche cachée)
    - Visualisation des 20 premières images mal classées
    - Projection PCA 2D des images MNIST

- "etude_learning_rate.py" : étude de l'influence du learning rate sur les
  trois modèles. Teste automatiquement la liste
  [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1] et calcule l'accuracy et le
  taux d'erreur (train et test). Produit dans "results/mnist/" :
    - learning_rate_study.csv
    - error_vs_learning_rate.png
    - accuracy_vs_learning_rate.png
    - learning_rate_comparison.png
  Conclusion : learning rate optimal autour de 0.01 (voir rapports/RAPPORT_MNIST.md).

- Dossier "params/" : paramètres entraînés des trois modèles, sauvegardés
  au format .npy après exécution de train(). Permet de recharger les modèles
  sans réentraîner (ce qui peut prendre plusieurs minutes).

  Résumé des taux d'erreur (test set) :
    Modèle linéaire     : 8.77%
    1 couche cachée     : 2.43%
    2 couches cachées   : 2.61%

============================================================
PARTIE 2 - CIFAR-10 (Classification d'images en couleur)
============================================================

Tous les fichiers se trouvent dans le dossier "CIFAR-10/".

- "preprocessing.py" : chargement du dataset CIFAR-10, conversion en
  niveaux de gris (formule 0.299R + 0.587G + 0.114B), reshape, normalisation
  et one-hot encoding. Exporte X_gris_train, X_gris_test, X_train (couleur),
  X_test (couleur), Y_train, Y_test.

- "CIFAR_10_gris_lineaire.py" : modèle linéaire sur images en niveaux de gris
  (vecteur 1024). Taux d'erreur : ~85.7% (test).

- "CIFAR_10_gris_couche_cachee.py" : réseau avec une couche cachée sur images
  en niveaux de gris. Mini-batch (batch_size=64). Taux d'erreur : ~68.5% (test).

- "CIFAR_10_couleur_lineaire.py" : modèle linéaire sur images couleur
  (vecteur 3072). Taux d'erreur : ~71.3% (test).

- "CIFAR_10_couleur_couche_cachee.py" : réseau avec une couche cachée sur
  images couleur. Mini-batch (batch_size=64). Taux d'erreur : ~55.8% (test).

- "convolution.py" : implémentation manuelle de la convolution 2D avec
  zero-padding et max-pooling. Application des 6 filtres du sujet (K1 flou,
  K2 netteté, K3-K6 détection de bords) sur une photo de chat.

- "CNN.py" : réseau de neurones convolutif (CNN) implémenté avec PyTorch
  (Option B du sujet). Architecture : Conv2d × 4 + MaxPool × 2 + Linear.
  Ajout d'un early stopping, de la sauvegarde du meilleur modèle
  (params/cnn_best.pth), des courbes de loss/accuracy et de la matrice de
  confusion (dans "results/cifar10/").

- "analyse_cifar.py" : compare les 5 modèles (les 4 modèles NumPy + le CNN).
  Réentraîne brièvement les modèles NumPy sur un sous-ensemble, lit les
  métriques du CNN et produit dans "results/cifar10/" :
    - metrics.json
    - model_comparison.png
  (voir rapports/RAPPORT_CIFAR10.md)

  Résumé des accuracy (test set, CIFAR-10) obtenues sur sous-ensemble :
    Modèle linéaire (gris)    : 17.7%
    Modèle linéaire (couleur) : 34.1%
    1 couche cachée (gris)    : 34.8%
    1 couche cachée (couleur) : 43.9%
    CNN (PyTorch, couleur)    : 48.6%


============================================================
PARTIE 3 - CBIS-DDSM (Classification de mammographies)
============================================================

Tous les fichiers se trouvent dans le dossier "CBIS-DDSM/".
Classification binaire : MALIGNANT (malin) contre BENIGN et
BENIGN_WITHOUT_CALLBACK (bénin).

Le CSV "mass_case_description_train_set.csv" est fourni dans le dossier CBIS-DDSM/.
Les images DICOM (disponibles sur TCIA - The Cancer Imaging Archive) doivent être
placées dans "CBIS-DDSM/images/" avec la structure d'origine :
   CBIS-DDSM/images/Mass-Training_P_00001_LEFT_CC/.../000000.dcm
   ...
Sans les images, le code bascule automatiquement sur des données SYNTHETIQUES.

- "preprocessing.py" : lecture du CSV, association image <-> label binaire,
  nettoyage des entrées invalides, redimensionnement (128x128 par défaut),
  normalisation, découpage train/validation/test (70/15/15), affichage de la
  répartition des classes et calcul du pos_weight (déséquilibre).

- "CNN_cbis.py" : CNN simple (3 blocs Conv+ReLU+MaxPool puis 2 couches denses).
  Adam, BCEWithLogitsLoss avec pos_weight (gestion du déséquilibre), early
  stopping et sauvegarde du meilleur modèle (params/cnn_cbis_best.pth).
  Calcule Accuracy, Precision, Recall, F1-score, ROC-AUC et sauvegarde
  "results/cbis_ddsm/metrics.json".

- "analyse_cbis.py" : produit la matrice de confusion, la courbe ROC et les
  courbes d'entraînement dans "results/cbis_ddsm/", et affiche une analyse
  médicale (faux positifs / faux négatifs). Voir rapports/RAPPORT_CBIS_DDSM.md.

  Résultats (test, données synthétiques) :
    Accuracy 0.956 | Precision 0.923 | Recall 0.923 | F1 0.923 | ROC-AUC 0.995


============================================================
RAPPORTS
============================================================

Le dossier "rapports/" contient :
- RAPPORT_MNIST.md      : étude du learning rate, graphiques, interprétation
- RAPPORT_CIFAR10.md    : comparaison des modèles, paramètres, overfitting
- RAPPORT_CBIS_DDSM.md  : prétraitement, architecture, métriques, analyse médicale


============================================================
INSTRUCTIONS POUR LANCER LE CODE
============================================================

1. Créer et activer l'environnement virtuel :
     python -m venv .venv
     .venv\Scripts\activate

2. Installer les dépendances :
     pip install -r requirements.txt

3. Lancer un fichier (exemple) :
     python MNIST/MNIST_lineaire.py

Note : les fichiers MNIST peuvent prendre plusieurs minutes à s'exécuter
sur les 60 000 images complètes. Pour tester rapidement, remplacer
X_train par X_train[:5000] dans l'appel à train().

Note : pour CNN.py, utiliser obligatoirement le Python du venv
(.venv\Scripts\python.exe) car PyTorch est installé dans le venv.

Note : les scripts qui sauvegardent dans "results/" et "params/" doivent être
lancés depuis la racine du projet (exemples) :
     python MNIST/etude_learning_rate.py
     python CIFAR-10/CNN.py
     python CIFAR-10/analyse_cifar.py
     python CBIS-DDSM/CNN_cbis.py
     python CBIS-DDSM/analyse_cbis.py
Bonjour, voici notre projet de ML (SM604) contenant la partie 1 dans son ensemble, et la partie 2 non finalisée et à peaufiner encore.

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
  Résumé des taux d'erreur (test set, CIFAR-10) :
                      | Niveaux de gris | Couleur
    Modèle linéaire   |     85.7%       |  71.3%
    1 couche cachée   |     68.5%       |  55.8%
    CNN (PyTorch)     |       -         |  ~46%*


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
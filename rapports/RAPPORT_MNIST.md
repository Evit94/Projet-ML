# Rapport MNIST — Étude de l'influence du learning rate

## Objectif

La partie MNIST (modèle linéaire, 1 couche cachée, 2 couches cachées) était déjà
terminée. On ajoute ici une **étude systématique de l'influence du learning rate**
sur les trois modèles, sans modifier leur architecture.

Le script `MNIST/etude_learning_rate.py` entraîne automatiquement les trois modèles
pour chaque learning rate de la liste :

```
[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
```

Pour garder un temps de calcul raisonnable (7 learning rates × 3 modèles),
l'entraînement se fait sur un sous-ensemble (10 000 images de train, 5 000 de test,
5 epochs). Les tendances restent parfaitement représentatives.

## Résultats

Les résultats complets sont dans `results/mnist/learning_rate_study.csv`.
Taux d'erreur (test) en fonction du learning rate :

| learning rate | Linéaire | 1 couche | 2 couches |
|---------------|----------|----------|-----------|
| 0.0001 | 19.90% | 42.06% | 88.58% |
| 0.0005 | 14.92% | 15.18% | 88.58% |
| 0.001  | 13.26% | 12.90% | 28.46% |
| 0.005  | 11.72% |  8.68% |  7.76% |
| **0.01**   | **11.62%** |  **7.74%** |  **6.88%** |
| 0.05   | 13.72% |  8.96% | 13.72% |
| 0.1    | 15.10% | 20.96% | 90.80% |

Graphiques produits dans `results/mnist/` :
- `error_vs_learning_rate.png` — erreur train/test vs learning rate (échelle log)
- `accuracy_vs_learning_rate.png` — accuracy train/test vs learning rate (échelle log)
- `learning_rate_comparison.png` — comparaison des trois modèles (erreur test)

## Interprétation

**Influence du learning rate.** Le learning rate contrôle la taille des pas de la
descente de gradient. On observe la forme classique en « U » : l'erreur est élevée
aux deux extrémités et minimale au milieu.

**Sous-apprentissage (learning rate trop faible).** Pour `lr = 0.0001`, les pas sont
trop petits : en 5 epochs les modèles n'ont pas le temps de converger. C'est
spectaculaire pour le réseau à 2 couches qui reste bloqué à ~88% d'erreur (proche du
hasard, 90%) : le signal de gradient est trop faible pour traverser deux couches
ReLU. Plus le réseau est profond, plus il est sensible à un learning rate trop petit.

**Convergence (learning rate optimal).** Le minimum est atteint autour de
`lr = 0.01` pour les trois modèles. À cette valeur, les modèles à couches cachées
exploitent pleinement leur capacité : 6.88% d'erreur test pour 2 couches, 7.74% pour
1 couche, contre 11.62% pour le modèle linéaire (limité par sa simplicité).

**Divergence (learning rate trop élevé).** Pour `lr = 0.1`, les pas sont trop grands :
l'optimisation « saute » par-dessus le minimum et oscille, voire diverge. Le réseau
à 2 couches repart à 90.80% d'erreur (divergence complète), le modèle 1 couche se
dégrade nettement (20.96%). Le modèle linéaire, plus stable, résiste un peu mieux.

**Learning rate optimal.** La valeur `lr = 0.01` (celle utilisée dans les modèles
d'origine) est confirmée comme le meilleur compromis pour les trois architectures.
La plage `[0.005, 0.05]` reste raisonnable pour les modèles peu profonds.

## Conclusion

Le learning rate est un hyperparamètre critique : un mauvais choix peut empêcher
totalement l'apprentissage (sous-apprentissage) ou le faire diverger. La sensibilité
augmente avec la profondeur du réseau — le modèle à 2 couches a la meilleure
performance au bon learning rate mais le plus mauvais comportement aux extrêmes.

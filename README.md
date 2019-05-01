### Jihane Sbaytti 
#### Polytech Paris-Sud
#### 2018/2019

<div align="center">

# RAPPORT PROJET MACHINE LEARNING

### Implémentation d’un Classifieur Bayésien pour la reconnaissance de chiffres

</div>

# 1. INTRODUCTION :

Dans ce projet, il est question de mettre en pratique les acquis du cours pour implémenter en Python3 un classifieur bayésien qui permet de reconnaître le contenu d’images. Les images choisies dans cette étude sont des photos de chiffres allant de 0 à 9. Nous voulons tester la performance de la classification bayésienne pour ce cas concret, et la comparer à d’autres méthodes de classification.

**Règle de décision de Bayes :**

Nous rappelons l’expression mathématique du théorème de Bayes qui prend en considération la probabilité à
priori d’apparition des individus des différentes classes et de leur distribution dans l’espace des descripteurs :
![](https://latex.codecogs.com/gif.latex?P\left(C_k\middle|&space;x\right)=\frac{{Pr}_{k\&space;}\cdot&space;f_{k\&space;}\left(x\right)}{\sum_{\dot{i}=1}^{C}{{Pr}_{i\&space;}\cdot&space;f_{i\&space;}\left(x\right)}}), où C est le nombre de classes.

Avec :

✓ P(C<sub>k</sub>|k): probabilité a posteriori que l’individu de coordonnées x appartienne à la classe k.

✓ Pr<sub>k</sub> : probabilité a priori que l’individu appartienne à la classe k.

✓ f<sub>k</sub> : densité de probabilité de x si la classe est k.

La règle de décision de Bayes consiste à choisir d’affecter l’individu à la classe dont la probabilité a posteriori (calculée par la formule de Bayes ou par tout autre méthode) est la plus grand. Cette décision **minimise le risque d’erreur de classification**.

# 2. DEVELOPPEMENT :

### 1.1 Données utilisées :

Pour ce projet, nous disposons de trois types de données chacun ayant un intérêt : apprentissage, développement et test(application). Les données sont des paires d’images et d’étiquettes.

Chaque image est une photo d’un chiffre de 0 à 9. Elle est de taille 24 x 32 pixels en 256 niveaux mais a été aplatie dans un tableau à 1 dimension de 24*32=768 pixels.

L’étiquette qui lui associée à une image est le chiffre qu’elle contient. Elle représente aussi la classe d’une image au vu du classifieur. C’est donc un nombre allant de 0 à 9.

| Type de données                  | Fichiers                                                                                          | Intérêt                                                         |
|----------------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
|    Données   d’apprentissage     |    trn_img.npy = tableau de 30000   images   trn_lbl.npy = tableau de 30000   étiquettes.         |    Calcul des moyennes et covariances   pour chaque classe      |
|    Données de   développement    |    dev_img.py = tableau de 5000 images   dev_lbl.npy = tableau de 5000   étiquettes.              |    Tester la pertinence de   l’apprentissage                    |
|    Données de   test             |    tst_img.npy = tableau d’images                                                                 |    Utilisation pratique                                         |



### 1.2 Implémentation : Méthodes et Algorithme

#### a) Phase d’apprentissage : 

Cette étape consiste à utiliser les données d’apprentissages pour apprendre à classifier les données et extraire les moyennes et covariances des classes. Pour l’implémenter, nous avons utilisé les étapes suivantes :

|    Méthodes                             |    Etapes de l’algorithme                                                                                                                                                                      |
|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    find_all_label_positions_in_array    |    Parcourir le tableau d'apprentissage contenant les étiquettes (y)   Trouver pour chaque classe/étiquette (de 0 à 9), toutes les positions   dans ce tableau qui lui correspondent.          |
|    get_images_at_positions              |    Récupérer les images qui correspondent à ces positions dans le   tableau d'apprentissage contenant les images                                                                               |
|    mean                                 |    Calculer la moyenne de toutes les images correspondant à une classe   (sera un vecteur de 768colonnes)                                                                                      |
|    covariance                           |    Calculer la matrice de covariance des images correspondant à une   étiquette (sera un vecteur de 768colonnes)                                                                               |
|    get_label_means_covariances          |    Produire un tableau contenant les 10 moyennes calculées de chaque classe   et leur 10 matrices de covariance.                                                                               |
|    compute_score                        |    Calculer le score d’une image par rapport à une classe                                                                                                                                      |

#### b) Phase de développement : 

Cette étape consiste à appliquer l’apprentissage sur les données de développement pour leur attribuer des étiquettes. Elle permet aussi d’évaluer le taux d’erreur de la classification.

|    Méthodes                                           |    Etapes de l’algorithme                                                                                                 |
|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
|    bayesian_classification   (Programme principal)    |    Classer toutes les images et renvoyer le tableau des étiquettes   attribuées.                                          |
|    error_rate                                         |    Comparer les étiquettes qu’attribue le classifieur avec les vraies   étiquettes des données de développement.          |

#### 3

## 1.3 Résultats :


#### a) Performance :

Le taux d’erreur est de **39,82%** et le temps d’exécution est de **21.5 0 secondes.**

Pour avoir une idée de la difficulté de classification, voici des exemples d’images mal classées :

N°1492 : ![](https://github.com/Takichiii/ImageClassifier/blob/master/res/img1.JPG).
N°491 : ![](https://github.com/Takichiii/ImageClassifier/blob/master/res/img2.JPG).
N°121 : ![](https://github.com/Takichiii/ImageClassifier/blob/master/res/img3.JPG).

|    Images               |    N°1492 |    N°491       |    N°121       |
|-------------------------|--------------|----------------|----------------|
|    Etiquette prédite    |    0         |    7           |    4           |
|    Vraie étiquette      |    6         |    2           |    9           |

Ces cas d’erreurs restent tolérables, étant donné que le 1 er exemple inclus une forme de 0 dans le chiffre 6, et que le 3 ème exemple est totalement fou et non discernable même pour un humain.

Il faut maintenant comparer cette performance avec d’autres algorithmes de classification disponible pour pouvoir l’évaluer.


#### b) Amélioration de la performance : application de l’algorithme CPA

L’algorithme d’analyse en composantes principales (ACP) permet de réduire la dimension des vecteurs d’images de 768 points à un vecteur de paramètres de plus petite taille. Il est disponible dans la librairie Scikit-Learn sklearn.decomposition.PCA.

Grâce à l’utilisation de cet algorithme sur les données d’entrée, nous pouvons réduire la dimension des vecteurs pour d’un côté, accélérer l’exécution, et d’un autre, améliorer la reconnaissance.

Le graphe suivant présente l’impact de la réduction de dimension des images sur le taux d'erreur du classifieur bayésien :

![alt text](https://github.com/Takichiii/ImageClassifier/blob/master/res/img4.JPG)
```
FIGURE 1 : PERFORMANCE DU CLASSIFIEUR BAYESIEN SELON LA TAILLE DES IMAGES REDUITES AVEC PCA
```

Nous remarquons que le meilleur taux de reconnaissance est atteint quand la taille des images produites par l’algorithme PCA est de **50 pixels**. Le taux d’erreur dans ce cas est de **14,26%,** ce qui représente une amélioration de performance **64 ,18%** par rapport à la classification sans réduction préalable.

#### c) Comparaison avec d’autres classifieurs :

Nous comparons la performance de notre classifieur bayésien avec les autres classifieurs vus en cours, et qui sont disponibles dans la librairie Scikit-Learn de Python.

- **SVC** = Support Vector Machine (Machines à vecteurs de support)
- **KNC2** = K Neighbors Classifier (Méthode des k plus proches voisins avec nombre de voisins =2)
- **KNC 10** = K Neighbors Classifier (nombre de voisins = 10 )
- **DTC** = Decision Tree Classifier (Arbre de décision)
- **GNB** = Gaussian Naive Bayes (Classifieur Bayésien Naïf)
- **LR** = Logistic Regression (Régression logistique)
- **LDA** = Linear Discriminant Analysis (Analyse discriminante linéaire)

N°1492 : ![alt text](https://github.com/Takichiii/ImageClassifier/blob/master/res/img5.JPG)
```
FIGURE 2 : TAUX D’ERREUR PAR ALGORITHME DE CLASSIFICATION
```

N°1492 : ![alt text](https://github.com/Takichiii/ImageClassifier/blob/master/res/img6.JPG)
```
FIGURE 3 : TEMPS D’EXECUTION DE DIFFERENTS ALGORITHMES DE CLASSIFICATION
```
|    Classifieur          	|    Bayésien    	|    SVC       	|    KNC2      	|    KNC10     	|    DTC       	|    GNB      	|    LR        	|    LDA       	|
|-------------------------	|----------------	|--------------	|--------------	|--------------	|--------------	|-------------	|--------------	|--------------	|
|    Temps d’exécution    	|    0.86        	|    138.31    	|    13.55     	|    14.24     	|    4.05      	|    0.11     	|    1.67      	|    0.33      	|
|    Taux d’erreur        	|    14,26%      	|    12,87%    	|    30,75%    	|    22,36%    	|    56,62%    	|    67,5%    	|    75,86%    	|    75,46%    	|


**Interprétation :**

✓L’algorithme le moins pertinent est la régression logistique.
✓L’algorithme le plus pertinent est l’algorithme des machines à vecteurs de support , suivi de près par le classifieur bayésien. Néanmoins, il est également l’algorithme le plus lent et son temps d’exécution est 160 fois plus grand que celui du classifieur bayésien.

## 1.4 Meilleur système et matrice de confusion :

Le système présentant le meilleur compromis en termes de vitesse et de taux de reconnaissance est finalement le **classifieur bayésien**. Malgré que le SVC donne 1,39% de plus de reconnaissances, il est beaucoup plus lent et ne sera pas adapté à des utilisations en temps réel par exemple où le classifieur bayésien peut être très intéressant.

La matrice de confusion pour la reconnaissance étudiée est alors :

N°1492 : ![alt text](https://github.com/Takichiii/ImageClassifier/tree/master/res/img7.JPG)
```
FIGURE 4 : MATRICE DE CONFUSION DU MEILLEUR SYSTEME POUR LA RECONNAISSANCE DE CHIFFRES
```
Dans cette matrice, les colonnes représentent les étiquettes attribuées par le classifieur et les lignes les vraies étiquettes. Un élément (i,j) de la matrice représente le nombre d’images classées. Nous disposons de 500 images pour chaque classe (données de développement).

✓ Dans la diagonale, nous pouvons lire le nombre d’images bien classées pour chaque classe (ie que l’étiquette attribuée = la vraie étiquette). Par exemple : 450 /500 images de 1 ont été bien prédites,
mais seulement 33 8 /500 images de 9 ont été correctement prédites. 
✓ En dehors de la diagonale, nous pouvons lire le nombre d’images mal classées. (ie que l’étiquette attribuée est différentes de la vraie étiquette). Par exemple : 1 a été confondu 6 fois avec 0.


**Interprétation :**

✓ Classement des chiffres du mieux reconnu au moins reconnu: 1 > 4 > 7 > 2 > 0 > 5 > 6 > 3 > 9 > 8
✓ Le plus grand nombre de confusion est de 45 , il montre que 0 a été pris pour un 9 45 fois par le classifieur.
✓ Le chiffre le mieux classé est 1 ( 450 /500 images) suivi par 7 (426/500).
✓ Le chiffre le moins confondu est 7 : il n’a été confondu avec 6 et 9 aucune fois.

# 3. CONCLUSION :

Le classifieur bayésien est l’une des méthodes les plus simples en apprentissage supervisé basée sur le théorème
de Bayes. Un avantage de cette méthode est la simplicité de programmation, la facilité d’estimation des
paramètres et sa rapidité (même sur de très grandes bases de données). Mais en pratique, il est peu utilisé par
les praticiens du Data Mining au détriment des méthodes traditionnelles que sont les arbres de décision ou les
régressions logistiques. Malgré ses avantages, son peu d’utilisation en pratique vient en partie du fait que ne
disposant pas d’un modèle explicite simple (l’explication de probabilité conditionnelle à priori), l’intérêt pratique
d’une telle technique est remis en question.



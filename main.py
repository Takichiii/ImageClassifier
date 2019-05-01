# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:53:37 2019

@author: JihaneSbaytti
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import time

"""
trnImages = tableau d'apprentissage de 30000 images. Chaque image est 
de taille 24 x 32 pixels en 256 niveaux mais a été aplatie dans un 
tableau à 1 dimension de 24*32=768 pixels

trnLabels = tableau d'apprentissage de 30000 étiquettes.Une étiquette est 
le numéro de la classe d'une image, il va de 0 à 9. 

devImages = tableau de developpement de 5000 images. Chaque image est 
de taille 24 x 32 pixels en 256 niveaux mais a été aplatie 
dans un tableau à 1 dimension de 24*32=768 pixels

devLabels = tableau de developpement de 5000 étiquettes. une étiquette 
est le numéro de la classe d'une image, elle va de 0 à 9. 
"""

def find_all_label_positions_in_array(labelArray, label):
    """
    Parcourir le tableau d'apprentissage contenant les étiquettes (y)
    Trouver pour chaque classe/étiquette (de 0 à 9), toutes les posisitions dans ce tableau qui lui correspondent
    """
    return [i for i, val in enumerate(labelArray) if val == label]

def get_images_at_positions(imagesArray, indices):
    """
    Récuperer les images qui correspondent à des indices dans le tableau d'apprentissage contenant les images
    """
    return imagesArray[indices]

def mean(labelImages) :
    """
    Faire la moyenne des images correspondant à une étiquette (sera un vecteur de 768colonnes)
    """
    mean = np.mean(labelImages, 0)
    return mean

def covariance(labelImages):
    covariance = np.cov(labelImages.T)
    det_log = np.linalg.slogdet(covariance) #computes the logarithm of the determinant rather than the determinant itself.
    covariance_inverse = np.linalg.inv(covariance)
    comp = ([det_log[0] * det_log[1], covariance_inverse])
    return comp


def get_label_means_covariances(trnImages, trnLabels) :
    """
    Produire deux tableaux de 10 éléments contenant les moyennes calculées  et la matrice de covariance pour chaque classe
    """
    meansImages = []
    covariances = []
    for label in range(0, 10):
        labelIndices = find_all_label_positions_in_array(trnLabels,label)
        labelImages = get_images_at_positions(trnImages, labelIndices)
        meansImages.append(mean(labelImages))
        covariances.append(covariance(labelImages))
    return meansImages, covariances


def compute_score(image, average_image, covariance):
    """
    Calculer le score d'une image de test avec une classe
    """
    diff = image - average_image
    score = -covariance[0] - diff.T @ covariance[1] @ diff
    return score

def guess_label_of_one_image(image, meansImages, covariances):
    """
    Calculer les scores d'une image avec chaque classe et renvoyer la classe qui a le meilleur score
    """
    max_score = -float("inf")
    score = 0
    label= 0
    for i in range(0,10):
        score = compute_score(image, meansImages[i], covariances[i])
        if max_score < score:
            max_score = score
            label = i
    return label

def bayesian_classification(devImages, trnImages, trnLabels):
    """
    Main program : devine la classe de chaque image de devImages
    """
    guessedLabels = []
    meansImages, covariances = get_label_means_covariances(trnImages, trnLabels)
    for i in range(0, len(devImages)):
        label = guess_label_of_one_image(devImages[i], meansImages, covariances)
        guessedLabels.append(label)
    return guessedLabels

def error_rate(guessedLabels, realLabels):
    """
    Tester la pertinence de la classification que fait notre programme
    """
    countBads = 0
    for i in range(0,len(guessedLabels)):
        if guessedLabels[i] != realLabels[i]:
            countBads = countBads + 1
    return countBads/len(realLabels)*100


def reduce_images_size(devImages, trnImages, size) :
    """
    Utiliser l'algorithme PCA pour réduire en amont la taille des vecteurs d'images en entrée de notre programme, de 768 points à un vecteur de paramètres de plus petite taille 
    """
    pca = PCA(size)
    pca.fit_transform(devImages)
    return pca.transform(devImages), pca.transform(trnImages)

def plot_bayesian_error_rate_by_pca_size(devImages, devLabels, trnImages, trnLabels):
    errorRates = []
    imageSizes = []
    for i in range(1, 20):
        trnImages_tmp, devImages_tmp = reduce_images_size(trnImages, devImages, i * 10)
        bayLabels = bayesian_classification(devImages_tmp, trnImages_tmp, trnLabels)
        errorRates.append(error_rate(bayLabels,devLabels))
        imageSizes.append(i * 10)
    plt.plot(imageSizes, errorRates, 'ro')
    plt.axis([0, 200, 0, 60])
    plt.xlabel('PCA Image size')
    plt.ylabel('Error rate (%)')
    plt.title('Variance in Bayesian Classifier Performance when applying PCA')
    plt.show()
    
def plot_classifiers_error_rate(devImages, devLabels, trnImages, trnLabels):
    """
    Tracer un graphe pour comparer les performances de notre classifieur bayésien avec tous les classifieurs de la librairie scikit
    """
    errorRates = []
    execTimes = []
    classifiers = [
            ('BAY', None),
            ('SVC', SVC(cache_size=200)),
            ('KNC2', KNeighborsClassifier(n_neighbors =2, n_jobs=-1)),
            ('KNC10', KNeighborsClassifier(n_neighbors =10, n_jobs=-1)),
            ('DTC', DecisionTreeClassifier()),
            ('GNB', GaussianNB()),
            ('LR', LogisticRegression()),
            ('LDA', LinearDiscriminantAnalysis()),
    ]
    trnImages, devImages = reduce_images_size(trnImages, devImages, 50)
    for name, classifier in classifiers:
        start_time = time.time()
        if name == "BAY":
            errorRates.append(error_rate(devLabels, bayesian_classification(devImages, trnImages, trnLabels)))
            execTimes.append(time.time() - start_time)
        else :
            if name == "SVC":
                trnImages_tmp = scale(trnImages)
                devImages_tmp = scale(devImages)
            classifier.fit(trnImages_tmp, trnLabels)
            errorRates.append(error_rate(classifier.predict(devImages_tmp), devLabels))
            execTimes.append(time.time() - start_time)
        print(name, "temps d'execution", execTimes[-1], "secondes, taux d'erreur : ", errorRates[-1], "%.")
    #plt.plot([i[0] for i in classifiers], errorRates, 'ro', label='error rate')
    plt.plot([i[0] for i in classifiers], execTimes, 'ro')
    #plt.axis([0, len(classifiers), 0, 100])
    plt.axis([0, len(classifiers), 0, 20])
    plt.xlabel('Classifier')
    plt.ylabel('Error rate (%)')
    plt.title('Error rate by classification algorithm')
    plt.ylabel('Execution time (s)')
    plt.title('Execution time by classification algorithm')
    plt.show()
    

X_trn = np.load('data/trn_img.npy') #devImages
y_trn = np.load('data/trn_lbl.npy') #devLabels
X_dev = np.load('data/dev_img.npy')     #trnImages
y_dev = np.load('data/dev_lbl.npy') #trnLabels
X_tst = np.load('data/tst_img.npy') #testImages

#Question 1
start_time = time.time()
y_bay = bayesian_classification(X_dev, X_trn, y_trn)
print("Question 1 : Taux d'erreur de notre classifieur bayésien avant réduction: ", error_rate(y_bay, y_dev), "% et temps d'exécution :", time.time() - start_time, "secondes.")


#Question 2
print("Question 2 : Impact de la réduction de dimension des images sur le taux d'erreur de notre classifieur bayésien :")
X_dev_reduced, X_trn_reduced = reduce_images_size(X_dev, X_trn,50)
y_bay = bayesian_classification(X_dev_reduced, X_trn_reduced, y_trn)
plot_bayesian_error_rate_by_pca_size(X_dev, y_dev, X_trn, y_trn)

#Question 3
print("Question 3 : Comparaison avec d'autres classifieurs ..")
plot_classifiers_error_rate(X_dev, y_dev, X_trn, y_trn)

#Sauvegarder le résultat du test 
print("Sauvegarde du résultat de test de notre classifieur bayésien dans tst_lbl.npy...")
X_tst_reduced, X_trn_reduced = reduce_images_size(X_tst, X_trn,20)
y_tst = bayesian_classification(X_tst, X_trn, y_trn)
np.save('tst_lbl.npy', y_tst)

#Matrice de confusion du meilleur système
print(confusion_matrix(y_dev, y_bay))

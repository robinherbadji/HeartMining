#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:32:45 2020

@author: yassine.sameh
"""


#==================== Import des modules necessaires==============
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import precision_score
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz



from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import sys, os



# Main definition - constants
menu_actions  = {}  
menu_actions_knn  = {}
menu_actions_bayes  = {}
  

# =======================
#     MENUS FUNCTIONS
# =======================

# Main menu
def main_menu():
    
    print("===========================================")
    print("Projet fouille de données,\n")
    print("Choisir une méthode de classification :\n")
    print("1. Plus proches voisins")
    print("2. Bayésienne")
    print("3. Arbre de décision")
    print("4. SVM")
    print("5. Réseau de Neurones")
    print("\n0. Quitter")
    choice = input(" >>  ")
    exec_menu(choice)

    return


# Back to main menu
def back():
    menu_actions['main_menu']()

# ============ KNN MENU ==========================

def knn_menu():
    
    print("===========================================")
    print("Méthode des K plus proches voisins,\n")
    print("Choisir une fonctionnalité :\n")
    print("1. K proches voisins")
    print("2. Graphique pour k voisins")
    print("9. Retour au menu principal")
    print("\n0. Quitter")
    choice = input(" >>  ")
    exec_menu_knn(choice)

    return

def exec_menu_knn(choice):
    ch = choice.lower()
    if ch == '':
        menu_actions_knn['knn_menu']()
    else:
        try:
            menu_actions_knn[ch]()
        except KeyError:
            print("Selection invalide, choisir une selection valide .\n")
            menu_actions_knn['knn_menu']()
    return

#------------ fonctions de classification knn -----------------
    

def Knn():
    
    #saisie du nombre de voisins
    k = eval(input("Saisir une valeur impaire valeur du nombre de voisins: "))
    
    if(k< 1):
        print(colored("Attention vous avez choisi un nombre inférieur ou égal à 0", 'red'))
        k = eval(input("Saisir une valeur impaire valeur du nombre de voisins: "))
    
    # lecture des donnees
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    
    # création des ensembles train / test
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train , y_train)
    predictions = knn.predict(X_test)
    
    # evaluation du classifieur
    cnf_matrix = confusion_matrix(predictions , y_test)
    print(cnf_matrix)
    print(classification_report(predictions , y_test))
    
    #retour au menu KNN
    menu_actions_knn['knn_menu']()

#fonction qui trace les précisions en fonction du nombre de voisins
def GraphKnn():
    
    k_accuracies= []
    error_rate = []
    
    nAccuracies = eval(input("Saisir le nombre de précisions de voisins que vous voulez afficher: "))
    if(nAccuracies<= 1):
        print(colored("Attention vous avez choisi un nombre inférieur ou égal à 1", 'red'))
        nAccuracies = eval(input("Saisir le nombre de précisions de voisins que vous voulez afficher: "))
            
    # lecture des donnees
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    
    # creation des ensembles train / test
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    
    #Conversion du nombre de precisions en nombre impair
    if(nAccuracies % 2 == 0):
        nAccuracies+=1
    
    for k in range(1,nAccuracies,2):

        # creation du classifieur
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train , y_train)
        predictions = knn.predict(X_test)
        
        accuracy= accuracy_score(y_test, predictions)
        error_rate.append(np.mean(predictions != y_test))
        k_accuracies.append(accuracy)
        
    #representation graphique 
    plt.plot(range(1,nAccuracies,2), k_accuracies, color='green', linestyle='solid', marker='o',
         markerfacecolor='green', markersize=5)
    plt.title('Precision des prédicitions en fonction du nombre de voisions')
    plt.xlabel('Valeur de k voisins')
    plt.ylabel('Precision')
    
    plt.show()
    
    #retour au menu KNN
    menu_actions_knn['knn_menu']()
    


#Fonctions relative au menu de KNN
    
def back_knn_menu():
    menu_actions['knn_menu']()

menu_actions_knn = {
    'knn_menu': knn_menu,
    '1': Knn,
    '2': GraphKnn,
    '9': back,
    '0': exit,
}

#============== BAYES MENU =======================
def bayes_menu():
    
    print("===========================================")
    print("Méthode Bayésienne,\n")
    print("Choisir une fonctionnalité :\n")
    print("1. Bayes")
    print("2. BayesVarSmoothing")
    print("3. Graphique Bayes")
    print("9. Retour au menu principal")
    print("\n0. Quitter")
    choice = input(" >>  ")
    exec_menu_bayes(choice)

    return

def exec_menu_bayes(choice):
    ch = choice.lower()
    if ch == '':
        menu_actions_bayes['bayes_menu']()
    else:
        try:
            menu_actions_bayes[ch]()
        except KeyError:
            print("Selection invalide, choisir une selection valide .\n")
            menu_actions_bayes['bayes_menu']()
    return

#------------ fonctions de classification bayésien -----------------

def Bayes():
    
    # lecture des donne ́es
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    # cr ́eation des ensembles train / test
    
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    
    #cre ́ation du classifieur
    gnb = GaussianNB()
    gnb.fit(X_train , y_train)
    predictions = gnb.predict(X_test)
    
    # e ́valuation du classifieur
    print(confusion_matrix(predictions ,y_test))
    print(classification_report(predictions , y_test))
    
    #Retour au menu Bayes
    menu_actions_bayes['bayes_menu']()
    
def BayesVarSmoothing():
    
    varSmoothing = eval(input("Choisir une variance ajoutée var_smoothing: "))
    
    if(varSmoothing< 1):
        print(colored("Attention vous avez choisi un nombre inférieur ou égal à 0", 'red'))
        varSmoothing = eval(input("Choisir une variance ajoutée var_smoothing: "))
    
    # lecture des donne ́es
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    # cr ́eation des ensembles train / test
    
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    
    #cre ́ation du classifieur
    gnb = GaussianNB(var_smoothing=varSmoothing)
    gnb.fit(X_train , y_train)
    predictions = gnb.predict(X_test)
    
    # e ́valuation du classifieur
    print(confusion_matrix(predictions ,y_test))
    print(classification_report(predictions , y_test))
    
    #Retour au menu Bayes
    menu_actions_bayes['bayes_menu']()

def GraphBayes():
    
    k_accuracies= []
    error_rate = []
    
    nAccuracies = eval(input("Saisir le nombre de variances que vous voulez afficher pour la précision: "))
    if(nAccuracies<= 1):
        print(colored("Attention vous avez choisi un nombre inférieur ou égal à 1", 'red'))
        nAccuracies = eval(input("Saisir le nombre de variances que vous voulez afficher pour la précision: "))
            
    # lecture des donnees
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    
    # cr ́eation des ensembles train / test
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    
    for k in range(1,nAccuracies):
        
        # cr ́eation du classifieur
        gnb = GaussianNB(var_smoothing=k)
        gnb.fit(X_train , y_train)
        predictions = gnb.predict(X_test)
        
        accuracy= accuracy_score(y_test, predictions)
        error_rate.append(np.mean(predictions != y_test))
        k_accuracies.append(accuracy)
        
        
    plt.plot(range(1,nAccuracies), k_accuracies, color='green', linestyle='solid', marker='o',
         markerfacecolor='green', markersize=5)
    plt.title('Precision des prédicitions en fonction de la variance ajoutée')
    plt.xlabel('Variance ajoutée')
    plt.ylabel('Precision')
    
    plt.show()
    
    #Retour au menu Bayes
    menu_actions_bayes['bayes_menu']()

def back_bayes_menu():
    menu_actions['bayes_menu']()


menu_actions_bayes = {
    'bayes_menu': bayes_menu,
    '1': Bayes,
    '2': BayesVarSmoothing,
    '3': GraphBayes,
    '9': back,
    '0': exit,
}

#=============== Arbre de décision ====================

def DecisionTree():
    # Récupération du train et du test
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    
    #####################################
    # Séparation du X et y
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    
    #####################################
    # Classification
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    
    #####################################
    # Calcul des prédictions
    y_pred = clf.predict(X_test)
    
    #####################################
    # Calcul de la matrice de confusion
    cnf_matix = confusion_matrix(y_pred,y_test)
    print(cnf_matix)
    
    #####################################
    # Calcul de la précision
    accuracy = precision_score(y_test, y_pred, average='weighted')
    print(' DecisionTreeClassifier accuracy score : {} '.format(accuracy))
    
    #####################################
    # Exportation de l'arbre en pdf
    print("\nExporting to pdf ...")
    tree_name = "decision_tree"
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data)
    graph.render(tree_name)
    print("Exporting OK -> " + tree_name + ".pdf")
    
    menu_actions['main_menu']()


#================ SVM =================================
def SVM():
    
    # Récupération du train et du test
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    
    #########################
    # Séparation du X et y
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test = df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    
    
    # Normalisation des donnes
    
    scaling = MinMaxScaler(feature_range = (-1,1)).fit(X_train)
    X_train = scaling.transform (X_train) 
    X_test = scaling.transform (X_test)
    
    # Trouver un meilleure solution pour split : StratifiedShuffleSplit ci-dessus
    X_train , X_trash , y_train , y_trash = train_test_split (X_train, 
                                                              y_train,
                                                              train_size = 0.3,
                                                              stratify = y_train)
    
    
    # Création du classifieur
    svmClassifier = svm.SVC(kernel ='rbf', C =1E6 , gamma ='auto')
    svmClassifier.fit(X_train, y_train)
    predictions = svmClassifier.predict(X_test)
    
    # Evaluation du classifieur
    cnf_matrix = confusion_matrix(predictions, y_test)
    print(cnf_matrix )
    print(classification_report(predictions, y_test))
    
    menu_actions['main_menu']()

#=============== Réseau de Neurones ===================
def Neural():
    # lecture des donne ́es
    df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
    df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)
    
    # cr ́eation des ensembles train / test
    
    
    
    X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
    X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]
    
    #Normalisationd des données
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    #Normalisation des données d'entrainement et de test
    StandardScaler(copy=True, with_mean=True, with_std=True)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    # cr ́eation du classifieur de reseau de neurones multicouches
    perceptron = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,
                               activation = 'relu',solver='adam',random_state=1)
    perceptron.fit(X_train , y_train)
    
    predictions = perceptron.predict(X_test)
    # e ́valuation du classifieur
    cnf_matrix = confusion_matrix(predictions , y_test)
    print(cnf_matrix)
    print(classification_report(predictions , y_test))
    print('Accuracy: %.2f' % accuracy_score(y_test, predictions))

    menu_actions['main_menu']()





#======================== Menu ===============================================
# Executer le menu
def exec_menu(choice):
    
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Selection invalide, choisir une selection valide .\n")
            menu_actions['main_menu']()
    return


# Exit le programme
def exit():
    sys.exit()

# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': knn_menu,
    '2': bayes_menu,
    '3': DecisionTree,
    '4': SVM,
    '5': Neural,
    '9': back,
    '0': exit,
}



# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    # Launch main menu
    main_menu()
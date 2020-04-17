#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:37:27 2020

@author: yassine.sameh
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# lecture des donne ́es
df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)

# cr ́eation des ensembles train / test

X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]



def Bayes(k):

    #cre ́ation du classifieur
    gnb = GaussianNB(var_smoothing=k)
    gnb.fit(X_train , y_train)
    predictions = gnb.predict(X_test)
    
    # e ́valuation du classifieur
    print(confusion_matrix(predictions ,y_test))
    print(classification_report(predictions , y_test))
    
    
k_accuracies= []
error_rate = []

def autoBayes():
    for k in range(1,100):
        
        # cr ́eation du classifieur
        gnb = GaussianNB(var_smoothing=k)
        gnb.fit(X_train , y_train)
        predictions = gnb.predict(X_test)
        
        accuracy= accuracy_score(y_test, predictions)
        error_rate.append(np.mean(predictions != y_test))
        k_accuracies.append(accuracy)
        
        
    plt.plot(range(1,100), k_accuracies, color='green', linestyle='solid', marker='o',
         markerfacecolor='green', markersize=5)
    plt.title('Precision des prédicitions en fonction de la variance ajoutée')
    plt.xlabel('Variance ajoutée')
    plt.ylabel('Precision')
    
    plt.show()
        
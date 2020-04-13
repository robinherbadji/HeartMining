#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:33:10 2020

@author: yassine.sameh
"""


import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# lecture des donne ́es
df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)

# cr ́eation des ensembles train / test

X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]


def KNN(k):
    # cr ́eation du classifieur
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train , y_train)
    
    predictions = knn.predict(X_test)
    # e ́valuation du classifieur
    cnf_matrix = confusion_matrix(predictions , y_test)
    print(cnf_matrix)
    print(classification_report(predictions , y_test))
    
    
    
def autoKNN():
    
    for k in range(1,22,2):
        
        # cr ́eation du classifieur
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train , y_train)
        
        predictions = knn.predict(X_test)
        # e ́valuation du classifieur
        cnf_matrix = confusion_matrix(predictions , y_test)
        

    
    



              
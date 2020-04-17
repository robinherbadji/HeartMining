#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:53:37 2020

@author: yassine.sameh
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# lecture des donne ́es
df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)

# cr ́eation des ensembles train / test

X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
X_test, y_test= df_test.iloc[:,0:-1], df_test.iloc[:,-1]


def NN():
    # cr ́eation du classifieur
    perceptron = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    perceptron.fit(X_train , y_train)
    
    predictions = perceptron.predict(X_test)
    # e ́valuation du classifieur
    cnf_matrix = confusion_matrix(predictions , y_test)
    print(cnf_matrix)
    print(classification_report(predictions , y_test))
    print('Accuracy: %.2f' % accuracy_score(y_test, predictions))

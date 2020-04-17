# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:31:59 2020
@author: Robin
"""

import pandas as pd
from sklearn import svm
from sklearn . metrics import confusion_matrix
from sklearn . metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import time as t


def display_runtime(start):
    run_time = t.time() - start
    hours = run_time // 3600
    minutes = (run_time % 3600) // 60
    seconds = (run_time % 3600) % 60
    result = ""
    if hours > 0:
        result = str(int(hours)) + "h "
    if minutes > 0:
        result += str(int(minutes)) + "min "
    result += str(int(seconds)) + "s "
    result += str(round((seconds-int(seconds))*1000)) + "ms"
    print('Elapsed Time : ', result)
    
    
    
# Démarrage du chronomètre
start_time = t.time()

# Récupération du train et du test
df_train = pd.read_csv('heartbeat/mitbih_train.csv', header=0)
df_test = pd.read_csv('heartbeat/mitbih_test.csv', header=0)


#########################
# Séparation du X et y
X_train, y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]
X_test, y_test = df_test.iloc[:,0:-1], df_test.iloc[:,-1]

"""
print(df_train.shape)
print(df_test.shape)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
"""

# Trouver un meilleure solution pour split : StratifiedShuffleSplit ci-dessus
X_train , X_trash , y_train , y_trash = train_test_split (X_train, 
                                                          y_train,
                                                          train_size = 0.3,
                                                          stratify = y_train)


# Création du classifieur
svm = svm.SVC(kernel ='rbf', C =1E6 , gamma ='auto')
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

# Evaluation du classifieur
cnf_matrix = confusion_matrix(predictions, y_test)
print(cnf_matrix )
print(classification_report(predictions, y_test))

display_runtime(start_time)
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:51:54 2020
@author: Robin
"""

import pandas as pd
from sklearn.metrics import precision_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import confusion_matrix 
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
print("Computing...")

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

display_runtime(start_time)


"""
def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)
"""
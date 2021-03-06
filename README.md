# HeartMining

Projet de Fouille de Donnée réalisé par MM. Yassine Sameh et Robin Herbadji - ECG Heartbeat Categorization Dataset https://www.kaggle.com/shayanfazeli/heartbeat
( Contraintes établies par Dr G. Forestier )

## Installation
- Recupérez le script Python HeartMining.py
- Recupérez le dataset sur le site kaggle : https://www.kaggle.com/shayanfazeli/heartbeat
- Vérifiez la présence des 2 fichiers mitbih_test.csv et mitbih_train.csv dans le dossier heartbeat

## Utilisation
A l'exécution du script HeartMining.py, vous aurez le choix entre différentes méthodes de classification du dataset:
- Plus proches voisins (KNN)
- Bayesienne
- Arbre de décision
- Support Vector Machine (SVM)
- Réseau de Neuronnes

## Context ECG Heartbeat Categorization Dataset
### Abstract
This dataset is composed of two collections of heartbeat signals derived from two famous datasets in heartbeat classification, the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database. The number of samples in both collections is large enough for training a deep neural network.

This dataset has been used in exploring heartbeat classification using deep neural network architectures, and observing some of the capabilities of transfer learning on it. The signals correspond to electrocardiogram (ECG) shapes of heartbeats for the normal case and the cases affected by different arrhythmias and myocardial infarction. These signals are preprocessed and segmented, with each segment corresponding to a heartbeat.

### Content
#### Arrhythmia Dataset
Number of Samples: 109446
Number of Categories: 5
Sampling Frequency: 125Hz
Data Source: Physionet's MIT-BIH Arrhythmia Dataset
Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
#### The PTB Diagnostic ECG Database
Number of Samples: 14552
Number of Categories: 2
Sampling Frequency: 125Hz
Data Source: Physionet's PTB Diagnostic Database
Remark: All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.

### Data Files
This dataset consists of a series of CSV files. Each of these CSV files contain a matrix, with each row representing an example in that portion of the dataset. The final element of each row denotes the class to which that example belongs.

### Acknowledgements
Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification: A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).

### Inspiration
Can you identify myocardial infarction?

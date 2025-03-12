from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn import preprocessing
import numpy as np

def load_data(DATASET):
    if DATASET == "IRIS":
        X, y = load_iris(return_X_y=True)
    elif DATASET == "WINE":
        X, y = load_wine(return_X_y=True)
    elif DATASET == "BREAST_CANCER":
        X, y = load_breast_cancer(return_X_y=True)
    elif DATASET == 'ECOLI':
        X = np.genfromtxt('datasets/ecoli/ecoli.data',delimiter=',', usecols = [1,2,3,4,5,6,7])
        y = np.genfromtxt('datasets/ecoli/ecoli.data',delimiter=',', usecols = [8], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
    elif DATASET == 'GLASS':
        X = np.genfromtxt('datasets/glass/glass.data',delimiter=',', usecols = [1,2,3,4,5,6,7,8,9])
        y = np.genfromtxt('datasets/glass/glass.data',delimiter=',', usecols = [10])
        y = preprocessing.LabelEncoder().fit(y).transform(y)
    elif DATASET == 'IONOSPHERE':
        X = np.genfromtxt('datasets/ionosphere/ionosphere.data',delimiter=',', usecols = [i for i in range(0, 34)])
        y = np.genfromtxt('datasets/ionosphere/ionosphere.data',delimiter=',', usecols = [34], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
    elif DATASET == 'LIVER':
        X = np.genfromtxt('datasets/liver/bupa.data',delimiter=',', usecols = [0, 1, 2, 3, 4, 5])
        y = np.genfromtxt('datasets/liver/bupa.data',delimiter=',', usecols = [6], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
    elif DATASET == 'SONAR':
        X = np.genfromtxt('datasets/sonar/sonar.data',delimiter=',', usecols = [i for i in range(0, 60)])
        y = np.genfromtxt('datasets/sonar/sonar.data',delimiter=',', usecols = [60], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)
    elif DATASET == "HEART":
        X = np.genfromtxt('datasets/heart/heart.data',delimiter=',', usecols = [0, 1, 3, 4, 5, 7, 9])
        y = np.genfromtxt('datasets/heart/heart.data',delimiter=',', usecols = [13], dtype=str)
        y[np.where(y != '0')[0]] = '1'
        y = preprocessing.LabelEncoder().fit(y).transform(y)
    elif DATASET == "PARKINSON":
        X = np.genfromtxt('datasets/parkinson/parkinsons.data',delimiter=',', usecols = [i for i in range(1, 17)] + [i for i in range(18, 24)])
        y = np.genfromtxt('datasets/parkinson/parkinsons.data',delimiter=',', usecols = [17], dtype=str)
        y = preprocessing.LabelEncoder().fit(y).transform(y)

    # Scale data
    X = preprocessing.StandardScaler().fit_transform(X)

    return X, y
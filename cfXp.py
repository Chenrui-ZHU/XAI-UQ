from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import shap
from sklearn import preprocessing
from scipy.stats import spearmanr
import numpy as np
import faiss
import uncertainties as unc

DATASET = "PARKINSON"

def __main__():
    compute_corr()

def attaigability_test():
    # Load and standardize dataset
    X, y = load_data()

    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Fit model
    model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Find neighbors
    nbrs = NearestNeighbors(n_neighbors=X_train.shape[0], algorithm='auto').fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)

    # al, ep = unc.eknn_uncertainties(X_train, y_train, X_test)
    # al, ep = unc.entropy_uncertainties(X_train, y_train, X_test)
    al, ep = unc.density_uncertainties(X_train, y_train, X_test)
    # al, ep = unc.centroids_uncertainties(X_train, y_train, X_test)
    
    uncertainties = al

    # Find counterfactuals
    counterfactual_indices = []
    for i in range(distances.shape[0]):
        counterfactual_indices.append(np.where(y_train[indices[i]] != y_pred[i])[0][0])

        # Visual plot
        # plt.scatter(X_train[:, 2], X_train[:, 3], s=15, c=y_train)
        # plt.scatter(X_test[i, 2], X_test[i, 3], s=25, c="blue")
        # plt.scatter(X_train[indices[i, counterfactual_indices[i]], 2], X_train[indices[i, counterfactual_indices[i]], 3], s=25, c="green")
        # plt.show()

    counterfactual_distances = np.array([distances[i, counterfactual_indices[i]] for i in range(distances.shape[0])])

    sorted_indices = np.argsort(uncertainties)
    uncertainties = uncertainties[sorted_indices]
    counterfactual_distances = counterfactual_distances[sorted_indices]

    plt.scatter(uncertainties, counterfactual_distances, alpha=0.5, c='r')
    plt.xlabel("Aleatoric uncertainty", fontsize=18)
    plt.ylabel("Dissimilarity", fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def compute_corr():
    # Load and standardize dataset
    X, y = load_data()
    X = preprocessing.StandardScaler().fit_transform(X)

    iterations = 100

    global_uncertainties = []
    global_unrobustness = []

    for _ in range(iterations):
        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Fit model
        model = KNeighborsClassifier(n_neighbors=7, weights="distance")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Fast Neighbors search
        index = faiss.IndexFlatL2(X_train.shape[1]) 
        index.add(X_train)
        distances, indices = index.search(X_test, X_train.shape[0])
        distances = np.sqrt(distances)

        # al, ep = unc.eknn_uncertainties(X_train, y_train, X_test)
        # al, ep = unc.entropy_uncertainties(X_train, y_train, X_test)
        # al, ep = unc.density_uncertainties(X_train, y_train, X_test)
        al, ep = unc.density_uncertainties(X_train, y_train, X_test)
        
        uncertainties = al

        # Find counterfactuals
        counterfactual_indices = []
        for i in range(distances.shape[0]):
            counterfactual_indices.append(np.where(y_train[indices[i]] != y_pred[i])[0][0])

        counterfactual_distances = np.array([distances[i, counterfactual_indices[i]] for i in range(distances.shape[0])])

        global_uncertainties.extend(uncertainties)
        global_unrobustness.extend(counterfactual_distances)


    stat, p_val = spearmanr(global_unrobustness, global_uncertainties)
    print(stat, p_val)

    plt.scatter(global_unrobustness, global_uncertainties, alpha=0.5, c='r')
    plt.xlabel("Aleatoric uncertainty", fontsize=18)
    plt.ylabel("Dissimilarity", fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def epistemic_reject():
    # Load and standardize dataset
    X, y = load_data()

    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Find neighbors
    nbrs = NearestNeighbors(n_neighbors=X_train.shape[0], algorithm='auto').fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)

    al, ep = unc.eknn_uncertainties(X_train, y_train, X_test)
    # al, ep = unc.entropy_uncertainties(X_train, y_train, X_test)
    # al, ep = unc.density_uncertainties(X_train, y_train, X_test)
    # al, ep = unc.centroids_uncertainties(X_train, y_train, X_test)
    
    uncertainties = ep

    density_list = []
    sorted_unc = np.sort(uncertainties)
    for i in range(X_test.shape[0]):
        density_list.append(len(np.where(uncertainties >= sorted_unc[i])[0]))

    plt.scatter(sorted_unc, density_list, alpha=0.5, c='b')
    plt.xlabel("Epistemic uncertainty", fontsize=18)
    plt.ylabel("Number of rejected explanations", fontsize=18)
    plt.tight_layout()
    plt.show()

    model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    model.fit(X_train, y_train)
    # Create a SHAP KernelExplainer with a sample of 20 training points
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 20))
    shap_values = explainer.shap_values(X_test)
    prediction = model.predict(X_test)
    # Each explanation is the SHAP vector corresponding to the predicted class
    explanations = np.array([shap_values[i][:, prediction[i]] for i in range(len(prediction))])

    indice = np.where(uncertainties == sorted_unc[-1])[0][0]

    feature_names = ["Refractive index", "Sodium", "Magnesium", "Aluminum", "Silicon", "Potassium", "Calcium", "Barium", "Iron"]
    shap_explanation = shap.Explanation(values=explanations[indice],base_values=np.mean(explanations[indice]),feature_names=feature_names)

    shap.plots.waterfall(shap_explanation)

    highest = np.argsort(np.absolute(explanations[indice]))[-2:]
    
    plt.scatter(X_train[:, highest[1]], X_train[:, highest[0]], c='black', s=15)
    plt.scatter(X_test[indice, highest[1]], X_test[indice, highest[0]], c='red', s=40)
    plt.xlabel(feature_names[highest[1]], fontsize=18)
    plt.ylabel(feature_names[highest[0]], fontsize=18)
    plt.tight_layout()
    plt.show()


def load_data():
    if DATASET == "IRIS":
        X, y = load_iris(return_X_y=True)
    elif DATASET == "WINE":
        X, y = load_wine(return_X_y=True)
    elif DATASET == "BREAST":
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

# __main__()
import numpy as np
from scipy.stats import entropy
from lib.deep_eknn import EKNN
# import lib.likelihood as density
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from lib.Uncertainty_ent import model_uncertainty
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize_scalar
import warnings

def deep_ensemble(X, y, X_test):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    models = []
    indices = np.arange(X.shape[0])
    num_classes = len(np.unique(y))

    al = np.zeros(X_test.shape[0])
    tot = np.zeros((X_test.shape[0],num_classes))

    for i in range(5):
        models.append(MLPClassifier(hidden_layer_sizes=(100, 20), activation='relu', solver='adam', max_iter=5000, random_state=i))
        models[i].fit(X[indices], y[indices])
        np.random.shuffle(indices)

        al += entropy(models[i].predict_proba(X_test), base=2, axis=1)
        tot += models[i].predict_proba(X_test)

    tot = tot / len(models)
    tot = entropy(tot, base=2, axis=1)
    al = al / len(models)

    ep = tot - al
    return al, ep

def centroids_uncertainties(X, y, X_test, length_scale=1):
    centroids = get_centroids(X, y)
    size = len(centroids)

    uncertainties = []
    for c in centroids:
        l2 = np.linalg.norm(X_test - centroids[c], ord=2, axis=1)
        l2 = (1/size) * (l2**2)
        l2 = l2 / (2 * length_scale)
        l2 = np.exp(-l2)
        uncertainties.append(l2)
    
    ep = 1 / np.max(uncertainties, axis=0)
    al = uncertainties / np.sum(uncertainties, axis=0)
    al = entropy(al, base=2, axis=0)

    return al, ep

def density_uncertainties(X, y ,X_test, k_neighbors = 7):
    ep, al = compute_epistemic_uncertainty(X, y, X_test, k_neighbors)
    return np.array(al), np.array(ep)

def entropy_uncertainties(X, y, X_test, min_samples_leaf=4):
    estimator = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
    estimator.fit(X, y)
    _, al, ep = model_uncertainty(estimator, X_test, X, y)

    return al, ep

def eknn_uncertainties(X, y, X_test, k_neighbors = 7):
    estimator = EKNN(len(np.unique(y)), k_neighbors)
    estimator.fit(X, y)

    al, ep = estimator.get_uncertainties(X_test)
    return al, ep

def get_centroids(X, y):
    unique_classes = np.unique(y)
    centroids = {}
    for cls in unique_classes:
        cls_positions = X[y == cls]
        centroids[cls] = np.mean(cls_positions, axis=0)
    return centroids

def compute_epistemic_uncertainty(X, y, X_test, k_neighbors):

    # Load model for classification
    classifier = KNeighborsClassifier(n_neighbors=k_neighbors, weights="distance")

    # Fit model according to the dataset
    classifier.fit(X, y)
    
    epistemic_list = []
    aleatoric_list = []

    # Compute epistemic uncertainty
    for i in range(X_test.shape[0]):
        dist, indices = classifier.kneighbors(np.array([X_test[i]]), k_neighbors)

        epistemic, aleatoric = compute_epistemic(dist[0], indices[0], np.array(y))
        epistemic_list.append(epistemic)
        aleatoric_list.append(aleatoric)

    return epistemic_list, aleatoric_list

def compute_epistemic(dist, indices, y):
    nb_classes = 2
    res = np.zeros(nb_classes)

    for i in range(indices.shape[0]):
        res[y[indices[i]]] += (1/dist[i])

    p = res[0]
    n = res[1]

    opt = minimize_scalar(f_objective_1, bounds=(0, 1), method='bounded', args=(p, n))
    pl1 = opt.x

    opt = minimize_scalar(f_objective_2, bounds=(0, 1), method='bounded', args=(p, n))
    pl2 = 1 - opt.x

    ue = min(pl1, pl2) - 0.5
    ua = 1 - max(pl1, pl2)

    return ue, ua

# Objective fuction used to compute epistemic uncertainty
def f_objective_1(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 2 * theta - 1

    res = min(left, right)

    return -res

# Objective fuction used to compute epistemic uncertainty
def f_objective_2(theta, p, n):
    left = ((theta**p) * (1-theta)**n) / (((p / (n+p))**p) * ((n / (n+p))**n))
    right = 1 - (2 * theta)
        
    res = min(left, right)

    return -res
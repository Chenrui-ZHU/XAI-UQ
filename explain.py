from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import entropy
import numpy as np
import shap

# Load Dataset
X, y = load_iris(return_X_y=True)

# Standardize data
X = preprocessing.StandardScaler().fit_transform(X)

uncertainty_list = []
robustness_list = []
for _ in range(100):
    # Split Train/Test sets (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    # Init and train model
    model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    model.fit(X_train, y_train)

    # SHAP
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 20))
    shap_values = explainer.shap_values(X_test)
    prediction = model.predict(X_test)
    explanations = np.array([shap_values[i,:,prediction[i]] for i in range(prediction.shape[0])])

    # Noise 
    noise = np.ones(X.shape[1]) / 10
    shap_values_noisy = explainer.shap_values(X_test + noise)
    prediction_noisy = model.predict(X_test + noise)
    explanations_noisy = np.array([shap_values_noisy[i,:,prediction_noisy[i]] for i in range(prediction_noisy.shape[0])])

    # Robustness (How NOT robust it is : distance between expl1 and expl2)
    robustness = np.linalg.norm(explanations - explanations_noisy, axis=1) 
    # print(robustness)

    # Uncertainty
    probas = model.predict_proba(X_test)
    uncertainties = entropy(probas, base=2, axis=1)

    # Sort certainties
    sorted_indices = np.argsort(uncertainties)

    uncertainty_list.append(uncertainties[sorted_indices])
    robustness_list.append(robustness[sorted_indices])

plt.plot(np.mean(uncertainty_list, axis=0), np.mean(robustness_list, axis=0))
plt.xlabel("Uncertainty")
plt.ylabel("Un-Robustness")
plt.title("Uncertainty vs Un-Robustness")
plt.savefig("figures/uncertainty_vs_unrobustness.png")
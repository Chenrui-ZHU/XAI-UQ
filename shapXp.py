from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import entropy
import numpy as np
import shap
import uncertainties as unc
import time
from math import sqrt
import data as dt
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import os

np.random.seed(42)

def attaigability_test(dataset):
    X, y = dt.load_data(dataset)

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

def process_iteration(X, y, n_neighbors, epsilon, uncertainty):
    nb_variables = X.shape[1]
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    model.fit(X_train, y_train)

    # Create a SHAP KernelExplainer with a sample of 20 training points
    explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_train, int(sqrt(X_train.shape[0]))*2))

    # Compute SHAP values for the test set and get explanations for each predicted class
    shap_values = explainer.shap_values(X_test)
    prediction = model.predict(X_test)
    # Correct extraction of the explanation vector for each test instance
    explanations = np.array([shap_values[i][:, prediction[i]] for i in range(len(prediction))])
    
    n_test = X_test.shape[0]
    # Generate random directions: shape (n_test, n_neighbors, nb_variables)
    directions = np.random.randn(n_test, n_neighbors, nb_variables)
    directions = directions / np.linalg.norm(directions, axis=2, keepdims=True)
    # Generate random radii with proper scaling: shape (n_test, n_neighbors)
    r = np.random.rand(n_test, n_neighbors) ** (1 / nb_variables) * epsilon
    # Compute neighbors: each test instance gets n_neighbors neighbors
    neighbors = X_test[:, None, :] + directions * r[:, :, None]
    # Reshape for batch processing: (n_test * n_neighbors, nb_variables)
    neighbors_flat = neighbors.reshape(-1, nb_variables)
    
    # Batch SHAP evaluation for all neighbor points
    shap_values_neighbors = explainer.shap_values(neighbors_flat)
    pred_neighbors_flat = model.predict(neighbors_flat)
    # Correctly extract the explanation vector for each neighbor
    explanations_neighbors_flat = np.array([
        shap_values_neighbors[i][:, pred_neighbors_flat[i]] for i in range(neighbors_flat.shape[0])
    ])
    # Reshape back to (n_test, n_neighbors, nb_variables)
    explanations_neighbors = explanations_neighbors_flat.reshape(n_test, n_neighbors, nb_variables)
    
    # Compute distances in explanations and inputs
    diff_explanations = np.linalg.norm(explanations[:, None, :] - explanations_neighbors, axis=2)
    diff_inputs = np.linalg.norm(X_test[:, None, :] - neighbors, axis=2)
    # Compute the ratio for each neighbor and take the maximum for each test instance
    ratios = diff_explanations / diff_inputs
    unrobust_values = np.max(ratios, axis=1)
    
    # Compute uncertainty
    if uncertainty == "entropy":
        al, ep = unc.entropy_uncertainties(X_train, y_train, X_test)
        uncertainties = al + ep
    if uncertainty == "density":
        al, ep = unc.density_uncertainties(X_train, y_train, X_test)
        uncertainties = al + ep
    if uncertainty == "eknn":
        al, ep = unc.eknn_uncertainties(X_train, y_train, X_test)
        uncertainties = al + ep
    if uncertainty == "centroids":
        al, ep = unc.centroids_uncertainties(X_train, y_train, X_test)
        uncertainties = al
    return uncertainties, unrobust_values

def robustness(uncertainty, dataset):
    X, y = dt.load_data(dataset)
    
    n_iterations = 5    # Number of iterations (increase for final experiments)
    n_neighbors = 30     # Number of neighbors to generate per test instance
    epsilon = sqrt(pow(0.1,2) * X.shape[1])        # Radius for generating neighbors
    
    print(f"sample size: {sqrt(X.shape[0]*0.7)*10}")

    # Run iterations in parallel using all available CPU cores
    start_time = time.time()

    # Uncomment the following lines to enable parallel processing
    # results = Parallel(n_jobs=-1)(
    #     delayed(process_iteration)(X, y, n_neighbors, epsilon, uncertainty)
    #     for _ in range(n_iterations)
    # )

    # For now, we will run the iterations sequentially
    results = []
    for _ in range(n_iterations):
        result = process_iteration(X, y, n_neighbors, epsilon, uncertainty)
        results.append(result)

    end_time = time.time()
    print(f"Total parallel execution time: {end_time - start_time:.2f} seconds.")

    corr_coef = []
    p_value = []
    for result in results:
        corr_coef_temp, p_value_temp = pearsonr(result[0], result[1])
        corr_coef.append(corr_coef_temp)
        p_value.append(p_value_temp)
    avg_corr_coef = np.mean(corr_coef)
    avg_p_value = np.mean(p_value)
    print("---SHAP robustness---")
    print(f"Average correlation coefficient ({dataset}_{uncertainty}): {avg_corr_coef}")
    print(f"Average p-value ({dataset}_{uncertainty}): {avg_p_value}")

    # Save correlation coefficients and p-values
    # corr = np.vstack((corr_coef, p_value)).T
    # np.save(f"output/shap/{uncertainty}/correlation_{dataset.lower()}_{uncertainty}.npy", corr)
    
    # Combine and sort by uncertainty
    all_results = []
    for res in results:
        temp = np.vstack((res[0], res[1])).T
        res_sorted = temp[np.argsort(temp[:, 0])]
        all_results.append(res_sorted)
        if n_iterations == 1:
            os.makedirs(f"figures/shap/{uncertainty}", exist_ok=True)
            plt.figure()
            plt.scatter(res_sorted[:, 0], res_sorted[:, 1], alpha=0.5, c='green')
            plt.xlabel(f"Total Uncertainty", fontsize=18)
            plt.ylabel("Un-Robustness", fontsize=18)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"figures/shap/{uncertainty}/{dataset.lower()}_{uncertainty}.png")
            plt.close()

    # Save all results
    # all_results = np.array(all_results)
    # np.save(f"output/shap/{uncertainty}/data_{dataset.lower()}_{uncertainty}.npy", all_results)

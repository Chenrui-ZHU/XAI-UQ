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
from scipy.stats import pearsonr, combine_pvalues
from joblib import Parallel, delayed
import os
import faiss
import gc

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

def process_iteration(X, y, n_neighbors, epsilon, uncertainty, n_iterations):
    nb_variables = X.shape[1]
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 41 + n_iterations)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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
        uncertainties = al
    if uncertainty == "density":
        al, ep = unc.density_uncertainties(X_train, y_train, X_test)
        uncertainties = al
    if uncertainty == "eknn":
        al, ep = unc.eknn_uncertainties(X_train, y_train, X_test)
        uncertainties = al
    if uncertainty == "centroids":
        al, ep = unc.centroids_uncertainties(X_train, y_train, X_test)
        uncertainties = al
    if uncertainty == "deep_ensemble":
        al, ep = unc.deep_ensemble(X_train, y_train, X_test)
        uncertainties = al

    del explainer
    gc.collect()

    return uncertainties, unrobust_values, ep, X_test

def robustness(uncertainty, dataset):
    X, y = dt.load_data(dataset)
    
    n_iterations = 5    # Number of iterations (increase for final experiments)
    n_neighbors = 30     # Number of neighbors to generate per test instance
    epsilon = sqrt(pow(0.1,2) * X.shape[1])        # Radius for generating neighbors
    
    print(f"sample size: {sqrt(X.shape[0]*0.7)*10}")

    # Run iterations in parallel using all available CPU cores
    start_time = time.time()

    # Uncomment the following lines to enable parallel processing
    # with Parallel(n_jobs=-1, backend="loky") as parallel:
    #     results = parallel(
    #         delayed(process_iteration)(X, y, n_neighbors, epsilon, uncertainty, i)
    #         for i in range(n_iterations)
    #     )

    # Run the iterations sequentially
    results = []
    for i in range(n_iterations):
        result = process_iteration(X, y, n_neighbors, epsilon, uncertainty, i)
        results.append(result)

    end_time = time.time()
    print(f"Total parallel execution time: {end_time - start_time:.2f} seconds.")

    all_uncertainties = []
    all_robustness = []

    for result in results:
        al, robust, ep, X_test = result
        idx = np.argsort(ep)
        idx = idx[:int(len(X_test)*0.7)] 
        all_uncertainties.extend(al[idx])
        all_robustness.extend(robust[idx])
    corr, p_val = pearsonr(all_uncertainties, all_robustness)
    print("---SHAP robustness---")
    print(f"Overall correlation coefficient ({dataset}_{uncertainty}): {corr:.4f}")
    print(f"Overall p-value ({dataset}_{uncertainty}): {p_val:.4e}")

    # Save correlation coefficients and p-values
    # os.makedirs(f"output/shap/al/{uncertainty}", exist_ok=True)
    # corr = np.vstack((corr, p_val)).T
    # np.savetxt(f"output/shap/al/{uncertainty}/correlation_{dataset.lower()}_{uncertainty}_2.csv", corr, delimiter=",", fmt='%s')
    
    # Combine and sort by uncertainty
    all_results = []
    for res in results:
        temp = np.vstack((res[0], res[1])).T
        # res_sorted = temp[np.argsort(temp[:, 0])]
        res_sorted = temp
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
    # np.save(f"output/shap/al/{uncertainty}/data_{dataset.lower()}_{uncertainty}_2.npy", all_results)
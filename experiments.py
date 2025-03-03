from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import entropy
import numpy as np
import shap

def training_test():
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

def robustness_test():
    # Load and standardize the iris dataset
    X, y = load_iris(return_X_y=True)
    X = preprocessing.StandardScaler().fit_transform(X)
    nb_variables = X.shape[1]
    
    # Lists to collect uncertainty and robustness for each experiment
    global_uncertainties = []
    global_robustness = []
    
    n_iterations = 100 # Number of iterations
    n_neighbors = 30 # Number of neighbors to generate
    epsilon = 0.1

    # Repeat experiment 100 times for smoothing
    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = KNeighborsClassifier(n_neighbors=7, weights="distance")
        model.fit(X_train, y_train)

        # Create a SHAP KernelExplainer using a sample of training points (20)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 20))
        
        # Compute SHAP values for the test set
        shap_values = explainer.shap_values(X_test)
        # Get model predictions for test set
        prediction = model.predict(X_test)
        # Extract the explanation for each test instance corresponding to its predicted class
        explanations = np.array([shap_values[i][:, prediction[i]] for i in range(len(prediction))])
        
        # List to hold the robustness value for each test instance in the current split
        robust_values = []
        
        # For each test instance x_i, generate a neighborhood and compute the maximum ratio
        for i, x in enumerate(X_test):
            f_x = explanations[i]  # explanation for instance x_i
            
            # Generate n_neighbors points in the epsilon-ball around x
            directions = np.random.randn(n_neighbors, nb_variables)  # Gaussian directions
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # Normalize directions
            # Sample radii with a distribution that makes points uniformly distributed in the ball
            r = np.random.rand(n_neighbors) ** (1 / nb_variables) * epsilon
            # Create the neighbor points: each neighbor is x + direction * radius
            neighbors = x + directions * r[:, np.newaxis]
            
            # Compute SHAP values for the neighbors
            shap_values_neighbors = explainer.shap_values(neighbors)
            # Get predictions for the neighbors
            pred_neighbors = model.predict(neighbors)
            # For each neighbor, extract its explanation corresponding to its predicted class
            explanations_neighbors = np.array([
                shap_values_neighbors[pred_neighbors[j]][:, pred_neighbors[j]] for j in range(n_neighbors)
            ])
            
            # Calculate the difference between f(x_i) and each neighbor explanation f(x_j)
            diff_explanations = np.linalg.norm(f_x - explanations_neighbors, axis=1)
            # Calculate the distance between x_i and each neighbor x_j
            diff_inputs = np.linalg.norm(x - neighbors, axis=1)
            # Compute the ratio for each neighbor
            ratios = diff_explanations / diff_inputs
            # The robustness measure for x_i is the maximum ratio over its neighborhood
            robust_values.append(np.max(ratios))
        
        robust_values = np.array(robust_values)
        
        # Compute uncertainty for each test instance (Shannon entropy on prediction probabilities)
        probas = model.predict_proba(X_test)
        uncertainties = entropy(probas, base=2, axis=1)
        
        # Append the uncertainties and robustness values to the global lists
        global_uncertainties.append(uncertainties)
        global_robustness.append(robust_values)
    
    # Concatenate results from all iterations
    global_uncertainties = np.concatenate(global_uncertainties)  
    global_robustness = np.concatenate(global_robustness)       

    # Create a combined 2D array for sorting by uncertainty (first column: uncertainty, second: robustness)
    combined = np.vstack((global_uncertainties, global_robustness)).T
    # Sort the combined array by increasing uncertainty (first column)
    combined_sorted = combined[np.argsort(combined[:, 0])]
    
    # Split the total sorted data evenly into 20 groups.
    groups = np.array_split(combined_sorted, 20)
    
    # Compute the average uncertainty and robustness for each group
    avg_uncertainties = [group[:, 0].mean() for group in groups]
    avg_robustness = [group[:, 1].mean() for group in groups]
    
    # Plot the smoothed curve: Robustness as a function of Uncertainty
    plt.figure()
    plt.plot(avg_uncertainties, avg_robustness, marker='o', linestyle='-')
    plt.xlabel("Uncertainty (Entropy)")
    plt.ylabel("Un-Robustness (Max Ratio)")
    plt.title("Smoothed Curve: Un-Robustness vs. Uncertainty")
    plt.savefig("figures/robustness_vs_uncertainty_smoothed.png")
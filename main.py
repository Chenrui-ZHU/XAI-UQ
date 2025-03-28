import shapXp as exp
import data as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

if __name__ == "__main__":
    # exp.training_test()
    dateset_names = [
        # "ECOLI",
        # "GLASS",
        # "IONOSPHERE",
        # "BREAST_CANCER",
        # "HEART",
        # "IRIS",
        "SONAR",
        # "LIVER",
        # "WINE",
        ]
    uncertainties = [
        # "entropy",
        # "eknn",
        # "density",
        "centroids",
    ]
    for dataset_name in dateset_names:
        for uncertainty in uncertainties:
            exp.robustness_test(uncertainty, dataset_name)
    print("Done!")
 
    # data = np.load("output/data_breast_cancer_eknn.npy", allow_pickle=True)
    # print(data.shape)
    # X, y = dt.load_data("BREAST_CANCER")
    # print(X.shape)
    # plt.figure()
    # plt.scatter(data[:171, 0], data[:171, 1], alpha=0.5, c='green')
    # plt.xlabel(f"Aleatoric Uncertainty (eknn)")
    # plt.ylabel("Dissimilarity")
    # plt.title(f"Curve: Un-Robustness vs. Uncertainty ({5} iterations)")
    # plt.savefig(f"figures/AL+EP/dissimilarity_vs_uncertainty_breast_cancer_eknn.png")

    # correlations = []
    # p_values = []
    # for i in range(5):
    #     start_idx = i * 171
    #     end_idx = (i + 1) * 171
    #     corr, p_value = pearsonr(data[start_idx:end_idx, 0], data[start_idx:end_idx, 1])
    #     correlations.append(corr)
    #     p_values.append(p_value)

    # # Compute the mean Pearson correlation
    # mean_correlation = np.mean(correlations)
    # mean_p_value = np.mean(p_values)
    # print(f"Mean Pearson Correlation over 5 iterations: {mean_correlation}")    
    # print(f"Mean p-value over 5 iterations: {mean_p_value}")

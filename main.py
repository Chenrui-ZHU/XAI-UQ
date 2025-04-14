import shapXp as shap_exp
import cfXp as cf_exp
import data as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

if __name__ == "__main__":
    # exp.training_test()
    dateset_names = [
        # "BREAST_CANCER",
        # "ECOLI",
        # "GLASS",
        # "HEART",
        # "IONOSPHERE",
        # "IRIS",
        # "LIVER",
        "PARKINSON",
        # "SONAR",
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
            shap_exp.robustness_test(uncertainty, dataset_name)
            # cf_exp.uncertainty_test(uncertainty, dataset_name)

            # data = np.load(f"output/eknn_al+ep/data_{dataset_name.lower()}_eknn.npy", allow_pickle=True)
            # for i in range(5):
            #     subset = data[i:i + 1].reshape(-1, 2)
            #     plt.figure()
            #     plt.scatter(subset[:, 0], subset[:, 1], alpha=0.5, c='green')
            #     plt.xlabel("Total Uncertainty", fontsize=18)
            #     plt.ylabel("Un-Robustness", fontsize=18)
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.tight_layout()
            #     plt.savefig(f"figures/AL+EP/{uncertainty}/unrobustness_vs_uncertainty_{dataset_name.lower()}_{uncertainty}_{i+1}.png")
            #     plt.close()
    print("Done!")
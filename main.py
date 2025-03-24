import shapXp as exp
import data as dt
import numpy as np

if __name__ == "__main__":
    # exp.training_test()
    dateset_names = [
        # "BREAST_CANCER",
        "HEART",
        "IRIS",
        # "SONAR",
        "LIVER",
        "WINE",
        ]
    uncertainties = [
        # "entropy",
        "eknn",
        # "density",
        # "centroids",
    ]
    for dataset_name in dateset_names:
        for uncertainty in uncertainties:
            exp.robustness_test(uncertainty, dataset_name)
    print("Done!")
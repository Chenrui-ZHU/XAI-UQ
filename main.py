import shapXp as shap_exp
import cfXp as cf_exp
import argparse

DATASET_LIST = [
    "BREAST_CANCER",
    "ECOLI",
    "GLASS",
    "HEART",
    "IONOSPHERE",
    "IRIS",
    "LIVER",
    "PARKINSON",
    "SONAR",
    "WINE",
]
UNCERTAINTY_LIST = [
    # "entropy",
    # "eknn",
    # "density",
    # "centroids",
    "deep_ensemble",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run ML pipeline on one or all datasets.")
    parser.add_argument("--dataset", type=str, help="Specify a dataset name.")
    parser.add_argument("--all", action="store_true", help="Run on all datasets.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.all:
        for dataset_name in DATASET_LIST:
            for uncertainty in UNCERTAINTY_LIST:
                print(f"Running pipeline on dataset: {dataset_name}")
                shap_exp.robustness(uncertainty, dataset_name)
                # cf_exp.robustness(uncertainty, dataset_name)
    elif args.dataset:
        for uncertainty in UNCERTAINTY_LIST:
            print(f"Running pipeline on dataset: {args.dataset}")
            shap_exp.robustness(uncertainty, args.dataset)
            # cf_exp.robustness(uncertainty, args.dataset)
    else:
        print("Please specify either --dataset DATASET or --all")

if __name__ == "__main__":
    main()
    print("Done!")
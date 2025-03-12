import shapXp as exp
import data as dt
import numpy as np

if __name__ == "__main__":
    # exp.training_test()
    # exp.robustness_test("density", "BREAST_CANCER")
    X, y = dt.load_data("HEART")
    print("X.shape: ", X.shape)
    print(np.max(X, axis=0))
    print(np.min(X, axis=0))
    print("Done!")
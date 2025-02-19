from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y
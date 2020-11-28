import numpy as np

class SVM:
    def __init__(self, n, rate=0.01, margin=10000):
        self.n = n
        self.weights = np.zeros((1, 1))
        self.rate = rate
        self.margin = margin

    def predict(self):
        print(self.weights)


if __name__ == "__main__":
    SVM(100).predict()

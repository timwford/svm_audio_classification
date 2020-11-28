import numpy as np
import pandas as pd

class SVM(object):
    def __init__(self, n, rate=0.01, margin=10000):
        self.n = n
        self.weights = np.zeros((1, 1))
        self.rate = rate
        self.margin = margin

    def cost(self, points):
        distances = 1 - points[:, -1] * (np.dot(points[:, :-1], self.weights[1:]) + self.weights[0])
        distances[distances < 0] = 0
        loss = self.margin * np.mean(distances)

        return 1 / 2 * np.dot(self.weights, self.weights) + loss

    def cost_gradient(self, points):

        distance = 1 - (points[-1] * (np.dot(points[:-1], self.weights[1:]) + self.weights[0]))
        delta_weights = np.zeros(self.weights.shape).astype(np.float64)

        if distance < 0:
            delta_weights += self.weights
        else:
            delta_weights[1:] += self.weights[1:] - (self.margin * points[-1] * points[:-1]).astype(np.float64)
            delta_weights[0] += self.weights[0] - (self.margin * points[-1])

        return delta_weights

    def predict(self, data):
        last_cost = float("inf")
        self.weights = np.zeros(data.shape[1]).astype(np.float64)

        for _ in range(self.n):
            for ind, point in enumerate(data):
                ascent = self.cost_gradient(point)
                self.weights = self.weights - (self.rate * ascent)
            cost = self.cost(data)
            last_cost = cost

        print(f"Last cost: {last_cost}")


if __name__ == "__main__":
    # df = pd.read_csv("dataset.csv")

    df = pd.read_csv("Iris.csv")
    df['SepalLengthCm'] =df['SepalLengthCm'].astype(float)
    df['SepalWidthCm'] = df['SepalWidthCm'].astype(float)
    df['PetalLengthCm'] = df['PetalLengthCm'].astype(float)
    df['PetalWidthCm'] = df['PetalWidthCm'].astype(float)

    print(df.head())

    x = df.iloc[1:100, [2, 3, 4]].values
    x[:, 2] = np.where(x[:, 2] == 'Iris-setosa', -1, 1)
    xlabel_text = "Index 2"
    ylabel_text = "Index 3"

    #print(x)

    SVM(1000).predict(x)

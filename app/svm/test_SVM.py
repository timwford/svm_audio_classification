import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from svm import SVM


if __name__ == "__main__":
    df = pd.read_csv("Iris.csv")
    df = df.drop(df.index[list(range(100, 150))])
    df = df.drop(['Id'], axis=1)

    df['SepalLengthCm'] = df['SepalLengthCm'].astype(float)
    df['SepalWidthCm'] = df['SepalWidthCm'].astype(float)
    df['PetalLengthCm'] = df['PetalLengthCm'].astype(float)
    df['PetalWidthCm'] = df['PetalWidthCm'].astype(float)

    classes = np.where(df['Species'].to_numpy() == 'Iris-setosa', -1, 1)

    df = df.drop(['Species', 'SepalLengthCm', 'PetalLengthCm'], axis=1)

    f1 = df.iloc[:, 0].tolist()
    f2 = df.iloc[:, 1].tolist()
    data = np.array([f1, f2]).reshape((-1, 2))

    x = data

    svm = SVM.SVM(10000)
    svm.fit(data, classes)
    svm.print_weights()

    correct = 0
    total = len(data)
    for index, row in enumerate(data):
        y_pred = svm.w1 * row[0] + svm.w2 * row[1] + svm.b

        if np.sign(y_pred) == classes[index]:
            correct += 1

    print(f"Accuracy: {correct/total}")


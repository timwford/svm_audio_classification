from utilities.enums import WaterState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from svm.SVM import SVM

filename = "train_water.csv"

def train_drip_model(df) -> SVM:
    drip_df = df[df['classification'].isin([WaterState.DRIP.name, WaterState.OFF.name])]
    drip_shuffle_df = drip_df.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(drip_shuffle_df.loc[:, ['amplitude', 'peak_count']],
                                                        drip_shuffle_df['classification'], test_size=0.2, random_state=42)

    drip_data = np.array(list(zip(X_train['amplitude'], X_train['peak_count']))).reshape((-1, 2))
    drip_data_test = np.array(list(zip(X_test['amplitude'], X_test['peak_count']))).reshape((-1, 2))
    drip_classes = np.where(y_train.to_numpy() == WaterState.ON.name, -1, 1).reshape(-1)
    drip_classes_test = np.where(y_test.to_numpy() == WaterState.ON.name, -1, 1).reshape(-1)

    drip_model = SVM(10000)
    drip_model.fit(drip_data, drip_classes)

    correct = 0
    total = len(drip_data_test)
    for index, row in enumerate(drip_data_test):
        y_pred = drip_model.w1 * row[0] + drip_model.w2 * row[1] + drip_model.b

        if np.sign(y_pred) == drip_classes_test[index]:
            correct += 1

    print(f"Accuracy for Drip Model: {correct / total}")

    return drip_model

def train_on_model(df) -> SVM:
    on_df = df[df['classification'].isin([WaterState.ON.name, WaterState.OFF.name])]
    on_shuffle_df = on_df.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(on_shuffle_df.loc[:, ['amplitude', 'peak_count']],
                                                        on_shuffle_df['classification'], test_size=0.2, random_state=42)

    on_data = np.array(list(zip(X_train['amplitude'], X_train['peak_count']))).reshape((-1, 2))
    on_data_test = np.array(list(zip(X_test['amplitude'], X_test['peak_count']))).reshape((-1, 2))
    on_classes = np.where(y_train.to_numpy() == WaterState.ON.name, -1, 1).reshape(-1)
    on_classes_test = np.where(y_test.to_numpy() == WaterState.ON.name, -1, 1).reshape(-1)

    on_model = SVM(10000)
    on_model.fit(on_data, on_classes)

    correct = 0
    total = len(on_data_test)
    for index, row in enumerate(on_data_test):
        y_pred = on_model.w1 * row[0] + on_model.w2 * row[1] + on_model.b

        if np.sign(y_pred) == on_classes_test[index]:
            correct += 1

    print(f"Accuracy for On Model: {correct / total}")

    return on_model


if __name__ == "__main__":
    df = pd.read_csv('train_water.csv')

    off = df[df['classification'] == WaterState.OFF.name]
    drip = df[df['classification'] == WaterState.DRIP.name]
    on = df[df['classification'] == WaterState.ON.name]

    plt.scatter(x=np.sqrt(off['amplitude']), y=np.sqrt(off['peak_count']), c='red')
    plt.scatter(x=np.sqrt(drip['amplitude']), y=np.sqrt(drip['peak_count']), c='blue')
    plt.scatter(x=np.sqrt(on['amplitude']), y=np.sqrt(on['peak_count']), c='green')
    plt.show()







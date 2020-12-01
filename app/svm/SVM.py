class SVM:
    def __init__(self, epochs, rate=0.0001):
        self.epochs = epochs
        self.rate = rate
        self.current_epoch = 1

        self.w1 = 0
        self.w2 = 0
        self.b = 0

    def regularization(self) -> float:
        return 1 / self.current_epoch

    def predict(self, data, classes):

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            y_pred = classes * (self.w1 * data[:, 0] + self.w2 * data[:, 1] + self.b)

            m1_deriv = 0
            m2_deriv = 0
            b_deriv = 0

            for index, value in enumerate(y_pred):
                if value < 1:
                    m1_deriv += data[index, 0] * classes[index]
                    m2_deriv += data[index, 1] * classes[index]
                    b_deriv += classes[index]

            self.w1 += self.rate * (m1_deriv - 2 * self.regularization() * self.w1)
            self.w2 += self.rate * (m2_deriv - 2 * self.regularization() * self.w2)
            self.b += self.rate * (b_deriv - 2 * self.regularization() * self.b)

    def print_weights(self):
        print(f"w1: {self.w1} w2: {self.w2} B: {self.b}")


if __name__ == "__main__":
    print('oh god')

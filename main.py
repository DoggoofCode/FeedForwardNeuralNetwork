import numpy as np
from numpy import array as arr


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FeedForwardNeuralNetwork:
    def __init__(self, *args):
        self.layers = args[0]
        self.weights = []
        self.biases = []
        self.learning_rate = 0.1

        # Initialize the weights and biases
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i - 1]))
            self.biases.append(np.random.randn(self.layers[i]))

        # print(self.weights)

    def run(self, X):
        for i in range(1, len(self.layers) - 1):
            X = sigmoid(self.weights[i] @ X + self.biases[i])
        return X

    def ai_train(self, X, y):
        # Feedforward
        a = [X]
        for i in range(len(self.weights)):
            a.append(self.run(a[i]))

        # Backpropagation
        error = y - a[-1]
        deltas = [error * a[-1] * (1 - a[-1])]
        for i in range(len(a) - 2, 0, -1):
            deltas.append(self.weights[i].T @ deltas[-1] * a[i] * (1 - a[i]))
        deltas = deltas[::-1]

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * deltas[i] @ a[i].T
            self.biases[i] += self.learning_rate * deltas[i]

    def fit(self, X, y, *, epochs=1000):
        epochs = self.__dict__.get('epochs', epochs)
        for _ in range(epochs):
            for i in range(len(X)):
                try:
                    self.ai_train(X[i], y[i])
                except IndexError:
                    raise Exception("The number of rows in X and y must be equal")

def main():
    X = arr([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = arr([0, 1, 1, 0])

    model = FeedForwardNeuralNetwork([2, 2, 1])
    model.fit(X, y)
    y_prediction = model.run(X[0])
    print(f"Prediction: {y_prediction}, Real: {y[0]}")


if __name__ == '__main__':
    main()

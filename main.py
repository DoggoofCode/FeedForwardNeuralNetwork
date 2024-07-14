import numpy as np
from numpy import array as arr
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    # return (np.tanh(x)+1)/2
    # relu
    return np.maximum(0, x)

class FeedForwardNeuralNetwork:
    def __init__(self, *args, **kwargs):
        self.layers = args[0]
        self.weights = []
        self.biases = []
        self.learning_rate = 0.1
        self.X = kwargs.get('X', None)
        self.y = kwargs.get('y', None)

        # Initialize the weights and biases
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i - 1]))
            self.biases.append(np.random.randn(self.layers[i], 1))

        # print(self.weights)

    def CostFunction(self, X, y):
        y_pred = [self.run(x)[0][0] for x in X]
        y_true = [y_single for y_single in y]
        return np.mean([(y_pred_s - y_true_s) ** 2 for y_pred_s, y_true_s in zip(y_pred, y_true)])

    def run(self, X):
        X = X.reshape(-1, 1)
        for i in range(len(self.weights)):
            X = sigmoid(self.weights[i] @ X + self.biases[i])
        return X

    def ai_train(self, X, y):
        # Feedforward
        a = [X.reshape(-1, 1)]
        for i in range(len(self.weights)):
            a.append(sigmoid(self.weights[i] @ a[i] + self.biases[i]))

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
        score = []
        for _ in range(epochs):
            for i in range(len(X)):
                score.append(self.CostFunction(X, y))
                try:
                    self.ai_train(X[i], y[i])
                except IndexError:
                    raise Exception("The number of rows in X and y must be equal")

        return score


def main():
    X = arr([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = arr([0, 1, 1, 0])

    for i in range(10):
        model = FeedForwardNeuralNetwork([2, 2, 1])
        score = model.fit(X, y, epochs=10000)
        y_prediction = model.run(X[0])
        print(f"Prediction: {y_prediction}, Real: {y[0]}")
        # print score on matplotlib
        # plot the score on the graph labeled with its index
        plt.plot(score, label=f"Model {i}")
        # save image to the current directory
        print(i)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

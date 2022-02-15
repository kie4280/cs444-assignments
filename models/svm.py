"""Support Vector Machine (SVM) model."""

import numpy as np
from models import utils


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_size: int = 8):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.bs = batch_size

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        dot = np.dot(X_train, self.w)
        grads = np.zeros_like(self.w, dtype=np.float64)
        diffs = np.expand_dims(dot[range(X_train.shape[0]), y_train], -1) - dot
        target_mask = np.zeros((X_train.shape[0], self.n_class), dtype=int)
        target_mask[range(X_train.shape[0]), y_train] = 1
        # print(diffs.shape)
        # print(X_train.shape)
        indicators = diffs < 1
        w_correct = np.sum(indicators, axis=1) - 1

        for i in range(self.n_class):
            grads[:, i] = self.w[:, i] * self.reg_const / X_train.shape[0] + np.mean(
                (np.expand_dims((indicators[:, i] * (1-target_mask[:, i]) - target_mask[:, i] * w_correct), -1)
                 ) * X_train, 0).T

        # print(indicators.astype(int))

        return grads

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        # X_train = np.copy(X_train)
        # y_train = np.copy(y_train)
        self.w = np.random.rand(X_train.shape[1], self.n_class)
        # y_train = np.expand_dims(y_train, -1)
        X_train = X_train.astype(dtype=np.float64)

        for i in range(self.epochs):
            for X, y in utils.minibatch(X_train, y_train, self.bs):
                self.w = self.w - self.lr * self.calc_gradient(X, y)
            acc = utils.get_acc(self.predict(X_train), y_train)
            self.lr = self.lr / 2
            print("epoch {} acc: {}".format(i+1, acc))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        return np.argmax(np.dot(X_test, self.w), axis=1)

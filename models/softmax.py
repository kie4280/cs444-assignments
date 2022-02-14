"""Softmax model."""

import numpy as np
from models import utils


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float,
                 batch_size: int = 15,  temperature: float = 1.0):
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
        self.temp = temperature

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray
                      ) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me

        m1 = np.dot(X_train, self.w)
        m1 = m1 / self.temp
        m1 = m1 - m1.max()
        classes_exp = np.exp(m1)
        exp_sum = np.sum(classes_exp, 1) + 1e-16

        one_hot = np.zeros((X_train.shape[0], self.n_class), dtype=int)
        one_hot[range(X_train.shape[0]), y_train] = 1

        # regu = np.ones_like(one_hot, dtype=np.float64) * self.reg_const / (self.n_class - 1)
        # regu[range(X_train.shape[0]),y_train] = 1-self.reg_const

        new_w = np.zeros_like(self.w, dtype=np.float64)
        for i in range(self.n_class):
            s = classes_exp[:, i] / exp_sum
            s = np.expand_dims(s - one_hot[:, i], -1) * X_train
            new_w[:, i] = np.mean(s, 0)

        # print(new_w.shape)
        return new_w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = np.copy(X_train)
        y_train = np.copy(y_train)
        self.w = np.random.rand(X_train.shape[1], self.n_class)
        y_train = np.expand_dims(y_train, -1)
        X_train = X_train.astype(dtype=np.float64)

        for i in range(self.epochs):
            for X, y in utils.minibatch(X_train, y_train, self.bs):
                self.w = self.w - self.lr * self.calc_gradient(X, y)
                pass
        return

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

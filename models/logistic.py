"""Logistic regression model."""

import math
import numpy as np
import models.utils as utils


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float, batch_size: int = 15):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.bs = batch_size

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        x = 1/(1 + np.exp(-z))
        return x

    def _empirical_loss(self, X: np.ndarray, y: np.ndarray):
        loss = -np.log(self.sigmoid(np.dot(self.w, X)))
        return loss

    def _loss_backward(self, X: np.ndarray, y: np.ndarray):
        out = (self.sigmoid(y * np.dot(X, self.w)))
        grad_loss = -(1-out) * y * X
        avg_grad = np.sum(grad_loss, axis=0) / X.shape[0]
        return np.expand_dims(avg_grad, axis=-1)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = np.copy(X_train)
        y_train = np.copy(y_train)
        self.w = np.random.rand(X_train.shape[1], 1)
        y_train = np.expand_dims(y_train, -1)
        y_train[y_train == 0] = -1
        X_train = X_train.astype(dtype=np.float64)

        for i in range(self.epochs):
            for X, y in utils.minibatch(X_train, y_train, self.bs):
                self.w = self.w - self.lr * self._loss_backward(X, y)
                pass

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
        X_test = X_test.astype(np.float64)
        dot = np.dot(X_test, self.w)
        pred = self.sigmoid(dot)
        output = np.zeros(X_test.shape[0], dtype=int)
        output[pred[:, 0] > self.threshold] = 1
        return output

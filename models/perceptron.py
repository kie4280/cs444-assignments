"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.random.rand(X_train.shape[1],self.n_class)
        
        for ep in range(self.epochs):
            if (ep % 10 == 0) and (ep != 0):
                self.lr = self.lr / 10
            for i in range(y_train.shape[0]):
                X_values = X_train[i]
                X_values = X_train[i].reshape((X_values.shape[0],1))
                scores = np.dot(self.w.T , X_values)
                correct_score = scores[y_train[i]]
                for j in range(self.n_class):
                    if j == y_train[i]:
                        
                        count = 0
                        for k in range(scores.shape[0]):
                            if (scores[k] > correct_score):
                                count = count + 1
                                
                        self.w.T[j] = self.w.T[j] + self.lr * X_train[i] * count                             
                    else:
                        if scores[j] > correct_score:
                            self.w.T[j] = self.w.T[j] - self.lr * X_train[i]
                        
        return self.w

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
        predictions = np.dot(X_test, self.w)
        return np.argmax(predictions,axis = 1)

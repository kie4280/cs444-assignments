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

    def signum_array(self, array):
        array[array > 0] = 1
        array[array <= 0] = -1
        return array
    
    def learning(self, X_train, y_train, weights):
        for i in range(self.epochs):
            for xi, yi in zip(X_train, y_train):
                y_hat = self.signum_array(np.dot(xi, weights))
                gradient = xi * (y_hat - yi)
                gradient = gradient.reshape(gradient.shape[0], 1)
                weights = weights - (gradient * self.lr)
        return weights
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.zeros((self.n_class, X_train.shape[1]))
        y_train_copy = np.copy(y_train)
        y_train_copy = y_train_copy.reshape((X_train.shape[0], 1))

        for i in range(self.n_class):
            print("Training Classifier: ", i+1)
            i_train = np.copy(y_train_copy)
            for j in range(X_train.shape[0]):
                if i_train[j, 0] == i:
                    i_train[j, 0] = 1
                else:
                    i_train[j, 0] = -1
                    
            weights = np.zeros((X_train.shape[1], 1))
            weights = self.learning(X_train, i_train, weights)
            self.w[i, :] = weights.T            
        
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
        predicted_class = self.signum_array(np.dot(self.w, X_test.T))
        prediction = np.argmax(predicted_class, axis=0).T
    
        return prediction

    

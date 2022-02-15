import math
import numpy as np



def minibatch(X: np.ndarray, y: np.ndarray, batch_size: int = 5):
    if X.shape[0] != y.shape[0]:
        raise ValueError("Instance mismatch")
    l = np.random.permutation(X.shape[0])
    X, y = X[l], y[l]
    for i in range(math.floor(X.shape[0]/batch_size) -1):
        yield X[batch_size *i:batch_size * (i+1), :], y[batch_size *i:batch_size * (i+1)]
    
    if X.shape[0] % batch_size != 0:
        yield X[-batch_size:,:], y[-batch_size:]

def get_acc(pred, y_test):
    return np.sum(y_test == pred) / len(y_test) * 100


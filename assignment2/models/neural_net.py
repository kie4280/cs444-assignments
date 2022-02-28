"""Neural network model."""

from typing import Dict, Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.
    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me

        return np.matmul(X, W) + b

    def weight_grad(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        output = np.zeros((X.shape[0], W.shape[1], W.shape[0], W.shape[1]))
        for i in range(X.shape[0]):
            for j in range(W.shape[1]):
                output[i, j, :, j] = X[i, :]

        return output

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        output = np.where(X > 0, X, 0)
        return output

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        diag = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            diag[i, :, :] = np.diagflat(np.where(X[i] > 0, 1, 0))
        return diag

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        X = X - np.max(X, axis=1, keepdims=True)
        cate = np.exp(X)
        return cate / np.sum(cate, axis=1, keepdims=True)

    def softmax_grad(self, X: np.ndarray) -> np.ndarray:
        """
        unsure
        Compute the gradient over all samples
        """
        output = np.zeros(
            (X.shape[0], X.shape[1], X.shape[1]), dtype=np.float64)
        for i in range(X.shape[0]):
            s = X[i]
            s = np.expand_dims(s, -1)
            y = np.matmul(s, -s.T)
            output[i, :, :] = y + np.diagflat(s)

        return output

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs: Dict[str, np.ndarray] = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.

        self.outputs[str(0)] = X.copy()

        for layer in range(1, self.num_layers + 1):
            W: np.ndarray = self.params["W" + str(layer)]
            b: np.ndarray = self.params["b" + str(layer)]
            X = self.linear(W, X, b)
            if layer == self.num_layers:
                X = self.softmax(X)
            else:
                X = self.relu(X)
            self.outputs[str(layer)] = X

        return X

    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Note: both gradients and loss should include regularization.
        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients: Dict[str, np.ndarray] = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        loss = -np.log(self.outputs[str(self.num_layers)]
                       [range(y.shape[0]), y])
        # regularization
        loss = np.mean(loss) + reg * np.sum(
            np.array([np.sum(self.params["W" + str(i+1)] ** 2) for i in range(self.num_layers)]))


        activ = self.outputs[str(self.num_layers)]
        ll_W = self.params["W" + str(self.num_layers)]
        ll_b = self.params["b" + str(self.num_layers)]

        grad_CEL = np.zeros((y.shape[0], 1, activ.shape[1]))
        grad_CEL[range(y.shape[0]), 0, y] = -1 / \
            activ[range(y.shape[0]), y]
        activ_grad = self.softmax_grad(activ)  # unsure
        grad_W = self.weight_grad(ll_W, self.outputs[str(self.num_layers-1)])
        grad_x = ll_W.T

        ba = np.matmul(grad_CEL, activ_grad)
        # print(ll_W.shape, grad_CEL.shape, activ_grad.shape, grad_W.shape)
        # print(g.shape)
        # gw = np.zeros((y.shape[0], 1, grad_W.shape[2], grad_W.shape[3]))

        # for i in range(y.shape[0]):
        #     # print(g[i].shape, grad_W.shape)
        #     gw[i] = np.tensordot(ba[i], grad_W[i], axes=1)
        # gw = gw.squeeze(axis=1)

        gw = np.expand_dims(self.outputs[str(self.num_layers-1)], -1)
        gw = np.matmul(gw, ba)

        grad_upstream = np.matmul(ba, grad_x)
        # unsure, probably need a identiy matrix?
        grad_b = np.identity(ba.shape[2])
        self.gradients["W" + str(self.num_layers)] = np.mean(gw, axis=0)
        self.gradients["b" + str(self.num_layers)
                       ] = np.mean(np.matmul(ba, grad_b), axis=0)

        for layer in range(self.num_layers-1, 0, -1):

            activ = self.outputs[str(layer)]
            ll_W = self.params["W" + str(layer)]
            ll_b = self.params["b" + str(layer)]

            activ_grad = self.relu_grad(activ)  # unsure
            # grad_W = self.weight_grad(ll_W, self.outputs[str(layer-1)])
            grad_x = ll_W.T

            ba = np.matmul(grad_upstream, activ_grad)
            # print(ll_W.shape, grad_CEL.shape, activ_grad.shape, grad_W.shape)
            # print(g.shape)
            # gw = np.zeros((y.shape[0], 1, grad_W.shape[2], grad_W.shape[3]))

            # for i in range(y.shape[0]):
            #     # print(g[i].shape, grad_W.shape)
            #     gw[i] = np.tensordot(ba[i], grad_W[i], axes=1)
            # gw = gw.squeeze(axis=1)
            # print(ba.shape, gw.shape)

            gw = np.expand_dims(self.outputs[str(layer-1)], -1)
            gw = np.matmul(gw, ba)

            grad_upstream = np.matmul(ba, grad_x)
            # unsure, probably need a identiy matrix?
            grad_b = np.identity(ba.shape[2])
            self.gradients["W" + str(layer)] = np.mean(gw, axis=0)
            self.gradients["b" +
                           str(layer)] = np.mean(np.matmul(ba, grad_b), axis=0)

        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.

        if opt == 'SGD':
            for i in range(self.num_layers):
                W: np.ndarray = self.params["W" + str(i+1)]
                b: np.ndarray = self.params["b" + str(i+1)]
                self.params["W" + str(i+1)] = W - lr * \
                    self.gradients["W" + str(i+1)]
                self.params["b" + str(i+1)] = b - lr * \
                    self.gradients["b" + str(i+1)]

            pass

        elif opt == 'Adam':
            pass
        pass

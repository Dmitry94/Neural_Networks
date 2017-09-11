from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params["W1"] = np.random.normal(0.0, weight_scale,
                                             (input_dim, hidden_dim))
        self.params["b1"] = np.zeros((hidden_dim,))

        self.params["W2"] = np.random.normal(0.0, weight_scale,
                                             (hidden_dim, num_classes))
        self.params["b2"] = np.zeros((num_classes,))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        scores, cache1 = affine_relu_forward(
            X, self.params["W1"], self.params["b1"])
        scores, cache2 = affine_forward(
            scores, self.params["W2"], self.params["b2"])

        if y is None:
            return scores

        grads = {}
        loss, d = softmax_loss(scores, y)
        loss = loss + 0.5 * self.reg * (np.sum(self.params["W1"] ** 2) +
                                        np.sum(self.params["W2"] ** 2))
        d, grads["W2"], grads["b2"] = affine_backward(d, cache2)
        grads["W2"] = grads["W2"] + self.reg * self.params["W2"]

        d, grads["W1"], grads["b1"] = affine_relu_backward(d, cache1)
        grads["W1"] = grads["W1"] + self.reg * self.params["W1"]

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        self.params["W1"] = np.random.normal(0, weight_scale,
                                             (input_dim, hidden_dims[0]))
        self.params["b1"] = np.zeros((hidden_dims[0],))
        if self.use_batchnorm:
            self.params["gamma1"] = np.ones((hidden_dims[0],))
            self.params["beta1"] = np.zeros((hidden_dims[0],))
        for i in xrange(0, len(hidden_dims) - 1):
            self.params["W%d" % (i + 2)] = np.random.normal(
                0, weight_scale, (hidden_dims[i], hidden_dims[i + 1]))
            self.params["b%d" % (i + 2)] = np.zeros((hidden_dims[i + 1]))
            if self.use_batchnorm:
                self.params["gamma%d" % (i + 2)] = np.ones((hidden_dims[i + 1]))
                self.params["beta%d" % (i + 2)] = np.zeros((hidden_dims[i + 1]))

        self.params["W%d" % (self.num_layers)] = np.random.normal(
            0, weight_scale, (hidden_dims[self.num_layers - 2], num_classes))
        self.params["b%d" % (self.num_layers)] = np.zeros((num_classes))

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(
                                   self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        cur_in = X
        caches = []
        reg_loss = 0.0
        for i in xrange(self.num_layers - 1):
            cur_in, cur_cache = affine_relu_forward(
                cur_in, self.params["W" + str(i + 1)],
                self.params["b" + str(i + 1)])
            caches.append(cur_cache)
            if self.use_batchnorm:
                cur_in, cur_cache = batchnorm_forward(
                    cur_in, self.params["gamma" + str(i + 1)],
                    self.params["beta" + str(i + 1)], self.bn_params[i])
                caches.append(cur_cache)
            if self.use_dropout:
                cur_in, cur_cache = dropout_forward(cur_in, self.dropout_param)
                caches.append(cur_cache)
            reg_loss += (0.5 * self.reg *
                         (np.sum(self.params["W" + str(i + 1)] ** 2)))

        scores, fin_cache = affine_forward(
            cur_in, self.params["W" + str(self.num_layers)],
            self.params["b" + str(self.num_layers)])
        reg_loss += (0.5 * self.reg *
                     (np.sum(self.params["W" + str(self.num_layers)] ** 2)))
        caches.append(fin_cache)

        if mode == 'test':
            return scores

        grads = {}
        loss, d = softmax_loss(scores, y)
        loss = loss + reg_loss

        k = -1
        (d, grads["W%d" % self.num_layers],
         grads["b%d" % self.num_layers]) = affine_backward(
             d, caches[k])
        grads["W%d" % self.num_layers] += (
            self.reg * self.params["W%d" % self.num_layers])
        k -= 1

        for i in xrange(self.num_layers - 1, 0, -1):
            if self.use_dropout:
                d = dropout_backward(d, caches[k])
                k -= 1
            if self.use_batchnorm:
                (d, grads["gamma%d" % i],
                 grads["beta%d" % i]) = batchnorm_backward(d, caches[k])
                k -= 1
            (d, grads["W%d" % i], grads["b%d" % i]) = affine_relu_backward(
                d, caches[k])
            grads["W%d" % i] += self.reg * self.params["W%d" % i]
            k -= 1

        return loss, grads

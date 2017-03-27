"""
    Neural network class with backprop.
"""

import numpy as np

class BNN(object):
    """
        Neural network class with backprop.
    """
    def __init__(self, hidden_layers_sizes, learning_rate=1e-2, reg_lambda=1e-3):
        self.hl_sizes = hidden_layers_sizes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.weights = np.array([])

    def train(self, data, labels):
        """
            Train net.

            Parameters:
            -------
            data : array, where each row - sample
            labels : array, where each row - label
        """
        # Init
        sensors_count = data.shape[1]
        classes_count = len(np.unique(labels))
        self._init_weights(sensors_count, classes_count)

        # Forward pass
        layers_outs = self._forward_pass(data)
        scores = layers_outs[len(layers_outs) - 1]

        # Calculate prababilities
        maxs = np.amax(scores, axis=1, keepdims=True)
        scores -= maxs
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        return 0

    def _init_weights(self, sensors_count, classes_count):
        layers_count = len(self.hl_sizes)
        k = 0.01

        w_start = k * np.random.randn(sensors_count, self.hl_sizes[0])
        self.weights = [w_start]
        for i in xrange(1, layers_count):
            w_cur = k * np.random.randn(self.hl_sizes[i - 1], self.hl_sizes[i])
            self.weights.append(w_cur)
        w_end = k * np.random.randn(self.hl_sizes[layers_count - 1], classes_count)
        self.weights.append(w_end)

    def _forward_pass(self, data):
        outs = [data]
        for i in xrange(1, 1 + len(self.weights)):
            cur_out = np.dot(outs[i - 1], self.weights[i - 1])
            outs.append(cur_out)

        return outs
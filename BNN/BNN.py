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

    def train(self, data, labels, print_loss=False, max_iters=10000):
        """
            Train net.

            Parameters:
            -------
            data : array, where each row - sample
            labels : array, where each row - label
        """
        # Init
        num_examples = data.shape[0]
        sensors_count = data.shape[1]
        classes_count = len(np.unique(labels))
        self._init_weights(sensors_count, classes_count)

        for i in xrange(max_iters):
            # Forward pass
            layers_outs = self._forward_pass(data)
            scores = layers_outs[-1]

            # Calculate prababilities
            maxs = np.amax(scores, axis=1, keepdims=True)
            scores -= maxs
            probs = np.exp(scores)
            probs = probs / np.sum(probs, axis=1, keepdims=True)

            # Calculate and print loss
            if print_loss and i % 1000 == 0:
                corect_logprobs = -np.log(probs[range(num_examples), labels])
                data_loss = np.sum(corect_logprobs) / num_examples

                reg_loss = 0.0
                for j in xrange(len(self.weights)):
                    reg_loss += 0.5 * self.reg_lambda * np.sum(self.weights[j] ** 2)
                loss = data_loss + reg_loss
                print "On iteration %d: loss is %f" % (i, loss)

            # Compute the gradient on scores
            dscores = probs
            dscores[range(num_examples), labels] -= 1
            dscores /= num_examples

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
        # Input
        outs = [data]

        # Out from hidden layers
        for i in xrange(len(self.weights) - 1):
            cur_out = np.maximum(0, np.dot(outs[i], self.weights[i]))
            outs.append(cur_out)

        # Out from out layer
        end_out = np.dot(cur_out, self.weights[-1])
        outs.append(end_out)

        return outs
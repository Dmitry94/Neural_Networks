"""
    Neural network class with backprop.
"""

import numpy as np

class BNN(object):
    """
        Neural network class with backprop.
    """
    def __init__(self, hidden_layers_sizes, learning_rate=1e-0, reg_lambda=1e-3):
        self.hl_sizes = hidden_layers_sizes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.weights = np.array([])
        self.biases = np.array([])

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
            probs = layers_outs[-1]

            # Calculate and print loss
            if print_loss and i % 1000 == 0:
                corect_logprobs = -np.log(probs[range(num_examples), labels])
                data_loss = np.sum(corect_logprobs) / num_examples

                reg_loss = 0.0
                for j in xrange(len(self.weights)):
                    reg_loss += 0.5 * self.reg_lambda * np.sum(self.weights[j] ** 2)
                loss = data_loss + reg_loss
                print "On iteration %d: loss is %f" % (i, loss)

            # Compute the gradient on probs, out layer
            dlast = probs
            dlast[range(num_examples), labels] -= 1
            dlast /= num_examples

            # Backprop on hidden layers
            for j in xrange(len(self.weights) - 1, 0, -1):
                cur_dw = np.dot(layers_outs[j].T, dlast) + self.reg_lambda * self.weights[j]
                cur_db = np.sum(dlast, axis=0, keepdims=True)
                dlast = np.dot(dlast, self.weights[j].T)
                dlast[layers_outs[j] <= 0] = 0

                self.weights[j] += -self.learning_rate * cur_dw
                self.biases[j] += -self.learning_rate * cur_db

            # Backprop on IN-layer
            in_dw = np.dot(data.T, dlast) + self.reg_lambda * self.weights[0]
            in_db = np.sum(dlast, axis=0, keepdims=True)
            self.weights[0] += -self.learning_rate * in_dw
            self.biases[0] += -self.learning_rate * in_db

    def predict(self, data):
        """
            Inference net for predicting label for each data row.

            Parameters:
            -------
            data : array, where each row is a sample

            Returns:
            array of labels
        """
        layers_outs = self._forward_pass(data)
        probs = layers_outs[-1]
        labels = np.argmax(probs, axis=1)

        return labels

    def _init_weights(self, sensors_count, classes_count):
        layers_count = len(self.hl_sizes)
        k = 0.01

        # In layer weights
        w_start = k * np.random.randn(sensors_count, self.hl_sizes[0])
        b_start = np.zeros((1, self.hl_sizes[0]))
        self.weights = [w_start]
        self.biases = [b_start]

        # Hidden layers weights
        for i in xrange(1, layers_count):
            w_cur = k * np.random.randn(self.hl_sizes[i - 1], self.hl_sizes[i])
            b_cur = np.zeros((1, self.hl_sizes[i]))
            self.weights.append(w_cur)
            self.biases.append(b_cur)

        # Out layer weights
        w_end = k * np.random.randn(self.hl_sizes[-1], classes_count)
        b_end = np.zeros((1, classes_count))
        self.weights.append(w_end)
        self.biases.append(b_end)

    def _forward_pass(self, data):
        # Input
        outs = [data]

        # Out from hidden layers
        for i in xrange(len(self.weights) - 1):
            cur_out = np.maximum(0, np.dot(outs[i], self.weights[i]) + self.biases[i])
            outs.append(cur_out)

        # Out from out layer
        end_out = np.dot(cur_out, self.weights[-1]) + self.biases[-1]
        # Calculate prababilities
        maxs = np.amax(end_out, axis=1, keepdims=True)
        end_out -= maxs
        probs = np.exp(end_out)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        outs.append(probs)

        return outs

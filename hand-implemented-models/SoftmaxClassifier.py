# -*- coding: UTF-8 -*-

"""Linear Softmax Classifier."""

import numpy as np
import tqdm


def compute_cross_entropy(data, labels, weights, reg_lambda):
    """
    Compute cross-entropy loss value.

    Take data. Process into through one neuron.
    Compute loss using outs and GT labels.

    Args:
        data: np.array
            input data for model
        labels: np.array
            GT labels for data
        weights: np.array
            model's weights
        reg_lambda: float
            regularization coefficient

    Returns:
        float: loss value

    """
    response = np.dot(data, weights)
    maxs = np.amax(response, axis=1, keepdims=True)
    response -= maxs
    response = np.exp(response)
    response = response / np.sum(response, axis=1, keepdims=True)
    correct_responses = -np.log(response[range(data.shape[0]), labels])

    reg_loss = 0.5 * reg_lambda * np.sum(weights ** 2)
    return np.sum(correct_responses) / data.shape[0] + reg_loss


def compute_cross_entropy_gradient(data, labels, weights, reg_lambda):
    """
    Compute cross-entropy loss gradient.

    Take data. Process into through one neuron.
    Compute loss gradient for neuron's weights using outs and GT labels.

    Args:
        data: np.array
            input data for model
        labels: np.array
            GT labels for data
        weights: np.array
            model's weights
        reg_lambda: float
            regularization coefficient

    Returns:
        np.array: same shape as weights have, gradient for it
                    to minimize loss value

    """
    response = np.dot(data, weights)
    maxs = np.amax(response, axis=1, keepdims=True)
    response -= maxs
    response = np.exp(response)
    response = response / np.sum(response, axis=1, keepdims=True)

    response[range(data.shape[0]), labels] -= 1
    response /= data.shape[0]

    reg_loss = reg_lambda * np.sum(weights)
    return np.dot(data.T, response) + reg_loss


class SoftmaxClassifier(object):
    """Softmax classifier class."""

    def __init__(self, learning_rate=0.01, reg_lambda=0.01, batch_size=1024):
        """Perform initialization."""
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights = np.array([])

    def train(self, data, labels, max_iters=10000, log=True):
        """
        Train softmax classifier.

        Args:
            data: np.array
                [N, K] data for training
            labels: np.array
                [N, 1] labels for training
            max_iters: int
                max count of training process iterations
            log: bool
                if True, then log progress into console

        """
        if log:
            iters_range = tqdm.trange(max_iters)
            iters_range.set_description('Softmax classifier training...')
        else:
            iters_range = range(max_iters)

        classes_count = len(np.unique(labels))
        self.weights = 0.01 * np.random.randn(data.shape[1] + 1, classes_count)

        cur_batch_size = min(self.batch_size, data.shape[0])
        for i in iters_range:
            rand_idxs = [np.random.randint(0, data.shape[0])
                         for _ in range(cur_batch_size)]

            ones = np.ones((cur_batch_size, 1))
            data_batch = data[rand_idxs]
            data_batch = np.hstack((data_batch, ones))
            labels_batch = labels[rand_idxs]

            gradient = compute_cross_entropy_gradient(
                data_batch, labels_batch, self.weights, self.reg_lambda)
            self.weights += -self.learning_rate * gradient

    def predict(self, data):
        """
        Predict labels for data.

        Args:
            data: np.array
                input data, for which labels need to be predicted

        Returns:
            np.array: labels array for data

        """
        ones = np.ones((data.shape[0], 1))
        data = np.hstack((data, ones))
        response = np.dot(data, self.weights)
        response = np.argmax(response, axis=1)

        return response

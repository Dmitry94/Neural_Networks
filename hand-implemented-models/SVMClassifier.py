# -*- coding: UTF-8 -*-

"""SVM Classifier."""

import numpy as np
import tqdm


def compute_svm_loss(data, labels, weights, margin, reg_lambda):
    """
    Calculate SVM loss value.

    Take some data [N, K]. Pass it through model.
    Use got outs and GT data to calculate SVM loss.

    Args:
        data: np.array
            input data, [N, K]
        labels: np.array
            labels for inputs data
        weights: np.array
            model weights
        margin: int
            margin value for SVM loss
        reg_lambda: float
            l2 reg coefficient

    Returns:
        float: loss value

    """
    response = weights.dot(data.T).T
    row_idx = np.arange(labels.shape[0])

    labels_resps = response[row_idx, labels].reshape(labels.shape[0], 1)
    margins = response - labels_resps + margin
    margins[row_idx, labels] = 0
    margins = np.amax(margins, axis=1)

    reg_loss = np.sum(weights ** 2) * reg_lambda
    return np.sum(margins) / data.shape[0] + reg_loss


def compute_svm_loss_gradient(data, labels, weights, margin, reg_lambda):
    """
    Calculate SVM loss gradient.

    Take some data [N, K]. Pass it through model.
    Use got outs and GT data to calculate SVM loss gradients.

    Args:
        data: np.array
            input data, [N, K]
        labels: np.array
            labels for inputs data
        weights: np.array
            model weights
        margin: int
            margin value for SVM loss
        reg_lambda: float
            l2 reg coefficient

    Returns:
        np.array: gradients for weights

    """
    response = weights.dot(data.T).T
    row_idx = np.arange(labels.shape[0])

    labels_resps = response[row_idx, labels].reshape(labels.shape[0], 1)
    margins = response - labels_resps + margin
    margins[row_idx, labels] = 0

    gradient_koefs = np.zeros(margins.shape)
    mask = margins > 0
    gradient_koefs[mask] = 1
    gradient_koefs[row_idx, labels] = -1 * np.sum(gradient_koefs, axis=1)

    return gradient_koefs.T.dot(data) / data.shape[0] + reg_lambda * weights


class SVMClassifier(object):
    """SVM classifier class."""

    def __init__(self, learning_rate=0.01, reg_lambda=0.01, batch_size=1024,
                 margin=1):
        """Perform start init."""
        self.reg_lambda = reg_lambda
        self.margin = margin
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights = np.array([])

    def train(self, data, labels, max_iters=10000, log=True):
        """
        Train SVM classifier.

        Args:
            data: np.array
                input train data
            labels:  np.array
                labels for input data
            max_iters: int
                max count of train iterations
            log: bool
                if True, then log result

        """
        if log:
            iters_range = tqdm.trange(max_iters)
            iters_range.set_description('SVM classifier train...')
        else:
            iters_range = range(max_iters)

        classes_count = len(np.unique(labels))
        self.weights = np.random.random_sample((
            classes_count, data.shape[1] + 1))

        cur_batch_size = min(self.batch_size, data.shape[0])
        for _ in iters_range:
            rand_idxs = [np.random.randint(0, data.shape[0])
                         for _ in range(cur_batch_size)]

            ones = np.ones((cur_batch_size, 1))
            data_batch = data[rand_idxs]
            data_batch = np.hstack((data_batch, ones))
            labels_batch = labels[rand_idxs]

            gradient = compute_svm_loss_gradient(
                data_batch, labels_batch,
                self.weights, self.margin, self.reg_lambda)

            self.weights += -self.learning_rate * gradient

    def predict(self, data):
        """
        Predict labels for specified data.

        Args:
            data: np.array
                input data, for which is necessary to calculate labels

        Returns:
            np.array: labels array

        """
        ones = np.ones((data.shape[0], 1))
        data = np.hstack((data, ones))
        response = self.weights.dot(data.T).T
        response = np.argmax(response, axis=1)

        return response

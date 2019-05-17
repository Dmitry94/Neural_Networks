# -*- coding: UTF-8 -*-

"""Nearest Neighbor Classifier demonstration."""

import numpy as np
import tqdm


class NNClassifier(object):
    """Nearest neighbor classifier class."""

    def __init__(self):
        """Perform default initialization."""
        self.data = np.array([])
        self.labels = np.array([])

    def train(self, data, labels):
        """
        Train classifier.

        Args:
            data: np.array
                input data array
            labels: np.array
                gt output data array

        """
        self.data = np.array(data)
        self.labels = np.array(labels)

    def predict(self, data, k=1, metric='L1', log=True):
        """
        Predict labels for input data.

        Args:
            data: np.array
                input data, for which classifier must calculate labels
                [N, K]: N - count, K - one case array
            k: int
                optional argument, that define, how many neighbors to analyze
            metric: str
                optional argument, that define, which metric to use
                there are two possible variants: L1 or L2
            log: bool
                optional argument, it True, then print progress bar

        Returns:
            np.array: array of labels for data

        """
        samples_count = data.shape[0]
        predictions = np.zeros(samples_count, dtype=self.labels.dtype)

        if log:
            iters_range = tqdm.trange(samples_count)
            iters_range.set_description('Nearest neighbor predicting...')
        else:
            iters_range = range(samples_count)

        for i in iters_range:
            # calc distances
            if metric == 'L1':
                distances = np.sum(np.abs(self.data - data[i, :]), axis=1)
            elif metric == 'L2':
                distances = np.sqrt(np.sum((self.data - data[i, :]) ** 2,
                                           axis=1))
            else:
                raise RuntimeError('Wrong `metric` argument value: {}'.format(
                    metric))

            # analyze neighbors
            top_mins = np.argpartition(distances, k)
            unique, counts = np.unique(self.labels[top_mins[0: k]],
                                       return_counts=True)
            label = np.argmax(counts)

            predictions[i] = unique[label]

        return predictions

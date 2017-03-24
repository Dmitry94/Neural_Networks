"""
    Linear SVM Classifier demonstration.
 """

import numpy as np
import utils as ut

def svm_loss(data, labels, weights, margin, reg_lambda):
    """
        Calculates linear SVM loss.

        Parameters:
        -------
        data : np.array, where each row is a sample
        labels : np.array, where each element is a label for a sample
        weights : np.array, weight matrix for cost function
        margin : int, margin for loss calcs
        reg_lambda : float, regularization koef

        Returns:
            Loss = svm_loss + reg_loss.
    """
    response = weights.dot(data.T).T
    row_idx = np.arange(labels.shape[0])

    labels_resps = response[row_idx, labels].reshape(labels.shape[0], 1)
    margins = response - labels_resps + margin
    margins[row_idx, labels] = 0
    margins = np.amax(margins, axis=1)

    reg_loss = np.sum(weights ** 2) * reg_lambda
    return np.sum(margins) / data.shape[0] + reg_loss

def svm_loss_gradient(data, labels, weights, margin, reg_lambda):
    """
        Analitic calculation of the svm_loss gradient.

        Parameters:
        -------
        data : np.array, where each row is a sample
        labels : np.array, where each element is a label for a sample
        weights : np.array, weight matrix for cost function
        margin : int, margin for loss calcs
        reg_lambda : float, regularization koef

        Returns:
            Gradient vector
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

    reg_derivative = 2 * weights[labels]
    reg_loss = np.sum(reg_derivative.T, axis=1) * reg_lambda

    return gradient_koefs.T.dot(data) / data.shape[0] + reg_loss


class SVMLinearClassifier(object):
    """
        Linear SVM classifier.
    """
    def __init__(self, learning_rate=0.01, reg_lambda=0.01, batch_size=1024, margin=1):
        self.reg_lambda = reg_lambda
        self.margin = margin
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights = np.array([])

    def train(self, data, labels, max_iters=1000):
        """
            Train SVM linear classifier.

            Parameters:
            -------
            data : np.array, where each row is data sample
            labels: np.array, where each element is sample labels
            max_iters: max count of iterations for training
        """
        classes_count = len(np.unique(labels))
        self.weights = np.random.random_sample((classes_count, data.shape[1] + 1))

        for _ in xrange(max_iters):
            cur_batch_size = min(self.batch_size, data.shape[0])
            rand_idxs = [np.random.randint(0, data.shape[0]) for i in xrange(cur_batch_size)]

            ones = np.ones((cur_batch_size, 1))
            data_batch = data[rand_idxs]
            data_batch = np.hstack((data_batch, ones))
            labels_batch = labels[rand_idxs]

            # Numerical
            # gradient = ut.compute_gradient(lambda x: svm_loss(data_batch, labels_batch, x,
            #                                                   self.margin, self.reg_lambda),
            #                                self.weights)

            # Analitic
            gradient = svm_loss_gradient(data_batch, labels_batch,
                                         self.weights, self.margin, self.reg_lambda)

            self.weights += -self.learning_rate * gradient

    def predict(self, data):
        """
            Predict labels for data.

            Parameters:
            -------
            data : np.array, where each row is data sample

            Returns:
            labels : np.array, where each element it's a label for sample from data.
        """
        ones = np.ones((data.shape[0], 1))
        data = np.hstack((data, ones))
        response = self.weights.dot(data.T).T
        response = np.argmax(response, axis=1)

        return response

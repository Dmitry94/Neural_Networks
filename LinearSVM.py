"""
    Linear SVM Classifier demonstration.
 """

import time
import numpy as np
import ciraf as cr
import utils as ut

def svm_loss(data, labels, weights, margin):
    """
        Calculates linear SVM loss.

        Parameters:
        -------
        data : np.array, where each row is a sample
        labels : np.array, where each element is a label for a sample
        weights : np.array, weight matrix for cost function
        margin : np.array, margin for loss calcs

        Returns:
            Mean margin.
    """
    response = weights.dot(data.T).T
    margins = np.maximum(0, response - response[labels] + margin)
    margins[labels] = 0

    return np.sum(margins) / data.shape[0]


class SVMLinearClassifier(object):
    """
        Linear SVM classifier.
    """
    def __init__(self, learning_rate=0.01, reg_lambda=0.01, margin=1):
        self.reg_lambda = reg_lambda
        self.margin = margin
        self.learning_rate = learning_rate
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
        bestloss = float("inf")
        classes_count = len(np.unique(labels))
        bath_size = 256
        self.weights = np.random.random_sample((classes_count, data.shape[1]))

        for _ in xrange(max_iters):
            rand_idxs = [np.random.randint(0, data.shape[0]) for i in xrange(bath_size)]
            data_batch = data[rand_idxs]
            labels_batch = labels[rand_idxs]

            reg_loss = np.sum(self.weights ** 2) * self.reg_lambda

            gradient = ut.calculate_gradient(lambda x: reg_loss +
                                             svm_loss(data_batch, labels_batch, x, self.margin),
                                             self.weights)
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
        response = self.weights.dot(data.T).T
        response = np.argmax(response, axis=1)

        return response





if __name__ == "__main__":
    TRAIN_BATCHES, TEST_BATCH = cr.read_ciraf_10("content/ciraf/cifar-10-batches-py")

    classifier = SVMLinearClassifier()

    start = time.clock()
    classifier.train(TRAIN_BATCHES[0]['data'], np.array(TRAIN_BATCHES[0]['labels']))
    end = time.clock()
    print 'Training time = ', end - start

    start = time.clock()
    predictions = classifier.predict(TEST_BATCH['data'])
    end = time.clock()
    print 'Prediction time = ', end - start

    accuracy = np.mean(predictions == TEST_BATCH['labels'])
    print 'NN Accuracy = %f' % accuracy
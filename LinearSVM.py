"""
    Linear SVM Classifier demonstration.
 """

import time
import numpy as np
import ciraf as cr

class SVMLinearClassifier(object):
    """
        Linear SVM classifier.
    """
    def __init__(self, reg_lambda=0.5, margin=1):
        self.reg_lambda = reg_lambda
        self.margin = margin
        self.weights = np.array([])

    def __calc_loss(self, data, labels, weights):
        response = weights.dot(data.T).T
        margins = np.maximum(0, response - response[labels] + self.margin)
        margins[labels] = 0

        return np.sum(margins) / data.shape[0]

    def train(self, data, labels, max_iters=100):
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
        for num in xrange(max_iters):
            cur_weights = np.random.random_sample((classes_count, data.shape[1]))
            loss = self.__calc_loss(data, labels, cur_weights)
            if loss < bestloss:
                bestloss = loss
                self.weights = cur_weights

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
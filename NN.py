"""
    Nearest Neighbor Classifier demonstration.
"""

import time
import numpy as np
import utils as ut
import ciraf as cr

class NearestNeighborClassifier(object):
    """
        Classifier.
    """
    def __init__(self):
        """
            Default init.
        """
        self.data = np.array([])
        self.labels = np.array([])

    def train(self, train_batches):
        """
            Train classifier on batches array.

            Parameters:
            -------
            train_batches : list[dictionary]
                It's list of batches, each batch - dictionary.
                Each dictionary must contains data and labels.
        """
        for batch in train_batches:
            batch_data = batch['data']
            batch_labels = batch['labels']

            if self.data.shape != batch_data.shape:
                self.data = batch_data
                self.labels = np.array(batch_labels)
            else:
                self.data = np.concatenate((self.data, batch_data), axis=0)
                self.labels = np.concatenate((self.labels, batch_labels), axis=0)

    def predict(self, test):
        """
            Assign labels for new data.

            Parameters:
            -------
            test : np.array[uint8]
                Array on DxN elements, where each row - sample,
                for each we have to assign label.
        """
        samples_count = test.shape[0]
        predictions = np.zeros(samples_count, dtype=self.labels.dtype)

        for i in xrange(samples_count):
            distances = ut.calc_l1(test[i, :], self.data)
            min_index = np.argmin(distances)
            predictions[i] = self.labels[min_index]

        return predictions


if __name__ == "__main__":
    TRAIN_BATCHES, TEST_BATCH = cr.read_ciraf_10("content/ciraf/cifar-10-batches-py", 1)

    classifier = NearestNeighborClassifier()

    start = time.clock()
    classifier.train(TRAIN_BATCHES)
    end = time.clock()
    print 'Training time = ', end - start

    start = time.clock()
    predictions = classifier.predict(TEST_BATCH['data'])
    end = time.clock()
    print 'Prediction time = ', end - start

    accuracy = np.mean(predictions == TEST_BATCH['labels'])
    print 'NN Accuracy = %f' % accuracy
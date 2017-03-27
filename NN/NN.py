"""
    Nearest Neighbor Classifier demonstration.
"""

import sys
sys.path.append('..')

import time
import numpy as np
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

    def train(self, data, labels):
        """
            Train classifier on batches array.

            Parameters:
            -------
            data : np.array
                Data, where each row has it's label in labels.
            labels : np.array
                Labels for data.
        """
        self.data = data
        self.labels = labels

    def predict(self, test, k=1):
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
            # L1
            distances = np.sum(np.abs(self.data - test[i, :]), axis=1)
            #L2
            #distances = np.sqrt(np.sum((self.data - test[i, :]) ** 2, axis=1))

            top_mins = np.argpartition(distances, k)
            unique, counts = np.unique(self.labels[top_mins[0: k]], return_counts=True)
            label = np.argmax(counts)

            predictions[i] = unique[label]

        return predictions


if __name__ == "__main__":
    TRAIN_BATCHES, TEST_BATCH = cr.read_ciraf_10("../content/ciraf/cifar-10-batches-py")

    classifier = NearestNeighborClassifier()

    start = time.clock()
    classifier.train(TRAIN_BATCHES[0]['data'], np.array(TRAIN_BATCHES[0]['labels']))
    end = time.clock()
    print 'Training time = ', end - start

    start = time.clock()
    predictions = classifier.predict(TEST_BATCH['data'], 5)
    end = time.clock()
    print 'Prediction time = ', end - start

    accuracy = np.mean(predictions == TEST_BATCH['labels'])
    print 'NN Accuracy = %f' % accuracy

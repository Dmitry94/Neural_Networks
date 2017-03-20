"""
    Nearest Neighbor Classifier demonstration.
"""

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
            Every batch should contains 'data' and 'label' fields
        """
        for batch in train_batches:
            batch_data = batch['data']
            batch_labels = batch['labels']

            if self.data.shape != batch_data.shape:
                self.data = batch_data
                self.labels = batch_labels
            else:
                self.data = np.concatenate((self.data, batch_data), axis=0)
                self.labels = np.concatenate((self.labels, batch_labels), axis=0)



if __name__ == "__main__":
    TRAIN_BATCHES, TEST_BATCH = cr.read_ciraf_10("content/ciraf/cifar-10-batches-py")

    classifier = NearestNeighborClassifier()
    classifier.train(TRAIN_BATCHES)

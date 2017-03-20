"""
    Linear SVM Classifier demonstration.
"""

import time
import numpy as np
import ciraf as cr

class SVMLin(object):
    """
        Linear SVM classifier.
    """
    def __init__(self, reg_lambda, margin=1):
        self.reg_lambda = reg_lambda
        self.margin = margin
        self.weights = np.array([])


if __name__ == "__main__":
    TRAIN_BATCHES, TEST_BATCH = cr.read_ciraf_10("content/ciraf/cifar-10-batches-py", 1)
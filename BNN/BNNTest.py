import sys
sys.path.append('..')

import time
import numpy as np

import BNN
import utils


# Generate data
X, Y = utils.generate_spiral_data(100, 3)
X_test, Y_test = utils.generate_spiral_data(300, 3)

HIDDEN_LAYERS_SIZES = [100]
net = BNN.BNN(HIDDEN_LAYERS_SIZES)
net.train(X, Y, print_loss=True)
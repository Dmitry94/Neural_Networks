import sys
sys.path.append('..')

import time
import numpy as np

import BNN
import utils


# Generate data
X, Y = utils.generate_spiral_data(100, 5)
X_test, Y_test = utils.generate_spiral_data(300, 5)

HIDDEN_LAYERS_SIZES = [100]
net = BNN.BNN(HIDDEN_LAYERS_SIZES)
net.train(X, Y, print_loss=True)

net_predict = net.predict(X_test)
accuracy = np.mean(net_predict == Y_test)
print accuracy
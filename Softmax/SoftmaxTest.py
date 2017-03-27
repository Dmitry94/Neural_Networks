"""
    Softmax test

    DOESN'T WORK, OVERFLOW
"""

import sys
sys.path.append('..')

import time
import numpy as np
import ciraf as cr

import Softmax as sfmx
from sklearn import svm as skSVM


BATCH_SIZE = 64
LEARNING_RATE = 0.01
REG_LAMBDA = 0.01
train_batches, test_batch = cr.read_ciraf_10("../content/ciraf/cifar-10-batches-py")
classifier = sfmx.SoftmaxClassifier(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, reg_lambda=REG_LAMBDA)


# Merge batches
train_data = train_batches[0]['data']
train_labels = np.array(train_batches[0]['labels'])
for i in xrange(1, len(train_batches)):
    cur_data = train_batches[i]['data']
    cur_labels = np.array(train_batches[i]['labels'])

    train_data = np.concatenate((train_data, cur_data))
    train_labels = np.concatenate((train_labels, cur_labels))



# Training
start = time.clock()
classifier.train(train_batches[0]['data'], np.array(train_batches[0]['labels']))
end = time.clock()
print 'My Softmax Training time = ', end - start



# Predicting
start = time.clock()
predictions = classifier.predict(test_batch['data'])
end = time.clock()
print 'My Softmax Predicting time = ', end - start

accuracy = np.mean(predictions == test_batch['labels'])
print 'My Softmax Accuracy = %f' % accuracy



clf = skSVM.LinearSVC()

start = time.clock()
clf.fit(train_data[:BATCH_SIZE], train_labels[:BATCH_SIZE])
end = time.clock()
print 'Sklearn SVM Training time = ', end - start

start = time.clock()
pr = clf.predict(test_batch['data']) 
end = time.clock()
print 'Sklearn SVM Predicting time = ', end - start

acc = np.mean(pr == test_batch['labels'])
print 'Sklearn acc = ', acc


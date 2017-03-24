
# coding: utf-8

# In[1]:

import time
import numpy as np
import ciraf as cr
import utils as ut

import LinearSVM as lSVM
from sklearn import svm as skSVM


# In[2]:

BATCH_SIZE = 512
LEARNING_RATE = 0.01
REG_LAMBDA = 0.01


# In[3]:

train_batches, test_batch = cr.read_ciraf_10("content/ciraf/cifar-10-batches-py")


# In[4]:

classifier = lSVM.SVMLinearClassifier(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, reg_lambda=REG_LAMBDA)


# In[5]:

# Merge batches
train_data = train_batches[0]['data']
train_labels = np.array(train_batches[0]['labels'])
for i in xrange(1, len(train_batches)):
    cur_data = train_batches[i]['data']
    cur_labels = np.array(train_batches[i]['labels'])

    train_data = np.concatenate((train_data, cur_data))
    train_labels = np.concatenate((train_labels, cur_labels))


# In[6]:

# Training
start = time.clock()
classifier.train(train_batches[0]['data'], np.array(train_batches[0]['labels']))
end = time.clock()
print 'My SVM Training time = ', end - start


# In[7]:

# Predicting
start = time.clock()
predictions = classifier.predict(test_batch['data'])
end = time.clock()
print 'My SVM Predicting time = ', end - start


# In[8]:

accuracy = np.mean(predictions == test_batch['labels'])
print 'My SVM Accuracy = %f' % accuracy


# In[9]:

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


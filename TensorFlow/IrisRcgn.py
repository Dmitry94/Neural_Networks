"""
    Demonstration TensorFlow for Iris recognition.
"""

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_data/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_data/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
        f.write(raw)


if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
        f.write(raw)

train_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# Specify input
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
classificator = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=[10, 20, 10],
                                               n_classes=3,
                                               model_dir="iris_data/iris_model")

def get_train_data():
    x_train = tf.constant(train_set.data)
    y_train = tf.constant(train_set.target)

    return x_train, y_train

def get_test_data():
    x_test = tf.constant(test_set.data)
    y_test = tf.constant(test_set.target)

    return x_test, y_test

classificator.fit(input_fn=get_train_data, steps=2000)

accuracy = classificator.evaluate(input_fn=get_test_data,
                                  steps=1)['accuracy']
print 'Accuracy = %f' % accuracy
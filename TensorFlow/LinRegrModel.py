"""
    Demonstrating tf.contrib.learn.LinearRegressor
"""

import tensorflow as tf
import numpy as np

# Declare features
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]


# Classificator
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])


# Specify how many batches and what it's size
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4,
                                              num_epochs=1000)

# Learning
estimator.fit(input_fn=input_fn, steps=1000)


# Test on train data
estimator.evaluate(input_fn=input_fn)
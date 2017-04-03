"""
    Demonstrationg how to create custom model.
"""

import numpy as np
import tensorflow as tf


def custom_model(features, labels, mode):
    """
        Custom model exampl
    """
    # Define y
    W = tf.Variable([1], dtype=tf.float64)
    b = tf.Variable([1], dtype=tf.float64)
    y = W * features['x'] + b

    # Define loss
    loss = tf.reduce_sum(tf.square(y - labels))

    # Define training
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(1e-2)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y,
        loss=loss, train_op=train)


estimator = tf.contrib.learn.Estimator(model_fn=custom_model)

x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
print estimator.evaluate(input_fn=input_fn, steps=10)

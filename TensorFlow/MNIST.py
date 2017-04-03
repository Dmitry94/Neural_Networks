"""
    Demonstrating TensorFlow for MNIST solving.
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# None means that length can be any
# 784 i.e. images are 28x28 pixels and grayscale
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Define all
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
predictions = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predictions),
                                              reduction_indices=[1]))

lr = 1e-0/2.0
optimizer = tf.train.GradientDescentOptimizer(lr)
train_step = optimizer.minimize(cross_entropy)


# Run training
session = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in xrange(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y: batch_ys})


# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x: mnist.test.images,
                                       y: mnist.test.labels}))

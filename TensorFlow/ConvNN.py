"""
    TensorFlow ConvNet demonstration.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    """
        Inits weights with normal distr.
    """
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def init_biases(shape):
    """
        Inits biases with zeros.
    """
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
        Creates convolutial layer.
    """
    # strides[0] == strides[3]
    # strides[1], strides[2] - width, height
    # padding='SAME' means that will be add zero pixel
    #   for saving width and hight on the out
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
        Creates pool layer.
    """
    # ksize means window of max reducing
    # ksize[0] == ksize[3]
    # ksize[1], ksize[2] - width, height
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 1, 1, 1], padding='SAME')


session = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# None means that length can be any
# 784 i.e. images are 28x28 pixels and grayscale
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# Add dimension for channel and fake dim for conv layer
x_image = tf.reshape(x, [-1, 28, 28, 1])


# Creating layers
W_conv1 = init_weights([5, 5, 1, 32])
b_conv1 = init_biases([32])

W_conv2 = init_weights([5, 5, 32, 64])
b_conv2 = init_biases([64])

W_fc1 = init_weights([7 * 7 * 64 * 1024])
b_fc1 = init_biases([1024])

W_fc2 = init_weights([1024, 10])
b_fc2 = init_biases([10])

# Fwd pass
conv_out1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
pool1 = max_pool_2x2(conv_out1)

conv_out2 = tf.nn.relu(conv2d(conv_out1, W_conv2) + b_conv2)
pool2 = max_pool_2x2(conv_out2)

tf.reshape(pool2, [-1, 7 * 7 * 64])
fc_out1 = tf.nn.relu(tf.matmul(pool2, W_fc1) + b_fc1)

# Dropout prob
keep_prob = tf.placeholder(tf.float32)
fc_out1 = tf.nn.dropout(fc_out1, keep_prob)

predicts = tf.matmul(fc_out1, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicts))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(predicts, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

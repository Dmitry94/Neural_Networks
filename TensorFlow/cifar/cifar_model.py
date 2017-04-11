"""
    Cifar10 Network model.
    Contains loss, input_fns, train, inrefence.
"""

# pylint: disable=C0103
# pylint: disable=C0330

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cifar_input

import tensorflow as tf
slim = tf.contrib.slim

IMAGE_SIZE = cifar_input.IMAGE_SIZE
NUM_CLASSES = cifar_input.NUM_CLASSES
TRAIN_SIZE = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
TEST_SIZE = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST

# Learning rate params
INIT_LR = 0.1
LR_DECAY_FACTOR = 0.1
NUM_EPOCHS_PER_DECAY = 350


# The decay to use for the moving average.
MOV_AVG_DECAY = 0.999


# Basic model parameters.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../../content/ciraf/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")




def train_inputs():
    """
        Get train inputs
    """
    return cifar_input.get_cifar10_input(
        FLAGS.data_dir, FLAGS.batch_size, False)


def test_inputs():
    """
        Get test inputs
    """
    return cifar_input.get_cifar10_input(
        FLAGS.data_dir, FLAGS.batch_size, True)


def inference(images):
    """
        Builds cifar10 model.

        Parameters:
        -------
        images: input data

        Returns:
        -------
        Logits
    """
    with tf.device('/CPU:0'):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.05),
                            weights_regularizer=slim.l2_regularizer(0.0)):
            net = slim.conv2d(images, 64, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
            net = tf.reshape(net, [FLAGS.batch_size, -1])

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.04),
                            weights_regularizer=slim.l2_regularizer(0.004)):
            net = slim.fully_connected(net, 384, scope='fc1')
            net = slim.fully_connected(net, 192, scope='fc2')
            net = slim.fully_connected(net, NUM_CLASSES,
                        activation_fn=None,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 1/192.0),
                        weights_regularizer=slim.l2_regularizer(0.0))

    return net


def loss(labels, logits):
    """
        Calculates loss.

        Parameters:
        -------
        labels: true labels for images
        logits: inference out

        Returns:
        -------
        Total loss
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                        logits=logits, name='cross_entropy_per_sample')
    ce_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', ce_mean)

    # Calculate total loss with weights reg penalty
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """
        Calculates avg losses and summary them.
    """
    losses = tf.get_collection('losses')
    loss_avgs = tf.train.ExponentialMovingAverage(MOV_AVG_DECAY, name='Avg')
    loss_avgs_op = loss_avgs.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.name + '(raw)', l)
        tf.summary.scalar(l.name, loss_avgs.average(l))

    return loss_avgs_op


def train(total_loss, global_step):
    """
        Train Ciraf10 model.

        Parameters:
        -------
        total_loss: loss for the model
        global_step: counter for training steps

        Returns:
        -------
        training op
    """
    num_batches_per_epoch = TRAIN_SIZE / FLAGS.batch_size
    lr_decay_steps = NUM_EPOCHS_PER_DECAY * num_batches_per_epoch

    lr = tf.train.exponential_decay(INIT_LR, global_step, lr_decay_steps,
                                    LR_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('Learning rate', lr)

    loss_avgs_op = _add_loss_summaries(total_loss)
    # While loss_avgs_op doesn't calc, context won't begin calc
    with tf.control_dependencies([loss_avgs_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradients_op = opt.apply_gradients(grads, global_step)

    # Add in summary histograms for vars and it's grads
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/histogram', var)

    for var, grad in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name +
                    '/gradients_hist', grad)

    # Calculate moving average for vars
    variable_avgs = tf.train.ExponentialMovingAverage(MOV_AVG_DECAY,
                                                      global_step)
    variable_avgs_op = variable_avgs.apply(tf.trainable_variables())

    with tf.control_dependencies([variable_avgs_op, apply_gradients_op]):
        train_op = tf.no_op('Train')

    return train_op

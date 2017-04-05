"""
    Cifar10 Network model.
    Contains loss, input_fns, train, inrefence.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import sys
import cifar_input

import tensorflow as tf

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
tf.app.flags.DEFINE_string('data_dir', '../../content/ciraf/cifar-10-batches-py',
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



def _tensor_summary(tensor):
    """
        Generates summury for some tensor.
        Summary it's histogram and sparsity.
    """
    tensor_name = tensor.op.name
    tf.summary.histogram(tensor_name, tensor)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(tensor))

def _get_variable_on_cpu(name, shape, initializer):
    """
        Creates variable stored on CPU.

        Parameters:
        -------
        name: variable name
        shape: variable shape
        initializer: variable initializer

        Returns:
        -------
        variable on cpu
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, tf.float32, initializer)

    return var


def _get_variable_on_cpu_with_reg(name, shape, stddev, rl):
    """
        Creates variable stored on CPU.
        With some deviation and regularization penalty.

        Paramters:
        -------
        name: variable name
        shape: variable shape
        stddev: desired deviation
        rl: regularization multiplicator
    """
    var = _get_variable_on_cpu(name, shape,
                tf.truncated_normal_initializer(stddev=stddev,
                                                dtype=tf.float32))

    # Calculate reg penalty and add it to the graph collection
    if rl is not None:
        reg_penalty = tf.multiply(tf.nn.l2_loss(var), rl, name='reg_loss')
        tf.add_to_collection('losses', reg_penalty)

    return var
    
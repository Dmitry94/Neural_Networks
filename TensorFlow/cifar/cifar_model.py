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

    # Tensor sparsiry, i.e. zeros_count / all
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

    # Prefer this, because can be scaleble for multi GPU
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
    # When we will calculate loss, we will get it too.
    if rl is not None:
        reg_penalty = tf.multiply(tf.nn.l2_loss(var), rl, name='reg_loss')
        tf.add_to_collection('losses', reg_penalty)

    return var


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

    # FIRST CONVOLUTION LAYER
    with tf.name_scope('conv1') as scope:
        weights = _get_variable_on_cpu_with_reg('weights',
                        shape=[5, 5, 3, 64], stddev=5e-2, rl=0.0)
        biases = _get_variable_on_cpu('biases', shape=[64],
                        initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(images, weights,
                    strides=[1, 1, 1, 1], padding='SAME')
        conv1_out = tf.nn.bias_add(conv, biases)
        conv1_out = tf.nn.relu(conv1_out, name=scope.name)
        _tensor_summary(conv1_out)

    # FIRST POOL LAYER
    pool1_out = tf.nn.max_pool(conv1_out, ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1], padding='SAME',
                    name='pool1')

    # FIRST NORM LAYER
    norm1_out = tf.nn.lrn(pool1_out, 4, bias=1.0,
                          alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

    # SECOND CONVOLUTION LAYER
    with tf.name_scope('conv2') as scope:
        weights = _get_variable_on_cpu_with_reg('weights',
                        shape=[5, 5, 64, 64], stddev=5e-2, rl=0.0)
        biases = _get_variable_on_cpu('biases', shape=[64],
                        initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(norm1_out, weights,
                    strides=[1, 1, 1, 1], padding='SAME')
        conv2_out = tf.nn.bias_add(conv, biases)
        conv2_out = tf.nn.relu(conv2_out, name=scope.name)
        _tensor_summary(conv2_out)

    # SECOND NORM LAYER
    norm2_out = tf.nn.lrn(conv2_out, 4, bias=1.0,
                          alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')

    # SECOND POOL LAYER
    pool2_out = tf.nn.max_pool(norm2_out, ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1], padding='SAME',
                    name='pool2')

    # FIRST FULL CONNECTED LAYER
    with tf.name_scope('fc1') as scope:
        pool2_out = tf.reshape(pool2_out, [FLAGS.batch_size, -1])
        dim = pool2_out.get_shape()[1].value
        weights = _get_variable_on_cpu_with_reg('weights',
                        shape=[dim, 384], stddev=0.04, rl=0.004)
        biases = _get_variable_on_cpu('biases', [384],
                            tf.constant_initializer(0.1))

        fc1_out = tf.nn.relu(tf.matmul(pool2_out, weights) + biases,
                    name=scope.name)
        _tensor_summary(fc1_out)

    # SECOND FULL CONNECTED LAYER
    with tf.name_scope('fc2') as scope:
        weights = _get_variable_on_cpu_with_reg('weights',
                        shape=[384, 192], stddev=0.04, rl=0.004)
        biases = _get_variable_on_cpu('biases', [192],
                        initializer=tf.constant_initializer(0.1))

        fc2_out = tf.nn.relu(tf.matmul(fc1_out, weights) + biases,
                    name=scope.name)
        _tensor_summary(fc2_out)

    # LAST, OUT LAYER
    with tf.name_scope('softmax-linear') as scope:
        weights = _get_variable_on_cpu_with_reg('weights',
                        shape=[192, NUM_CLASSES], stddev=1/192.0,
                        rl=0.0)
        biases = _get_variable_on_cpu('biases', [NUM_CLASSES],
                    initializer=tf.constant_initializer(0.0))

        fc3_out = tf.add(tf.matmul(fc2_out, weights), biases,
                        name=scope.name)
        _tensor_summary(fc3_out)

    return fc3_out


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

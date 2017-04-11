"""
    Cifar10 Network model.
"""

import tensorflow as tf
slim = tf.contrib.slim

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

def inference(images, labels_count):
    """
        Builds cifar10 model.

        Parameters:
        -------
        images: input data
        labels_count: count of labels

        Returns:
        -------
        Logits
    """
    with tf.device('/CPU:0'):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.05),
                            weights_regularizer=slim.l2_regularizer(0.0)):
            net = slim.conv2d(images, 64, [5, 5], scope='conv1')
            _tensor_summary(net)
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm1')

            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            _tensor_summary(net)
            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')

            net = tf.reshape(net, [images.get_shape()[0].value, -1])

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.04),
                            weights_regularizer=slim.l2_regularizer(0.004)):
            net = slim.fully_connected(net, 384, scope='fc1')
            _tensor_summary(net)

            net = slim.fully_connected(net, 192, scope='fc2')
            _tensor_summary(net)


        net = slim.fully_connected(net, labels_count,
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(0.0, 1/192.0),
                    weights_regularizer=slim.l2_regularizer(0.0),
                    scope='softmax-linear')
        _tensor_summary(net)

    return net

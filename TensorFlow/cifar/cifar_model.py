"""
    Cifar10 Network model.
"""

# pylint: disable=C0103
# pylint: disable=C0330

from collections import namedtuple

import tensorflow as tf
slim = tf.contrib.slim


Conv2dParams = namedtuple('Conv2dParams', ['stride', 'ksize', 'filters_count'],
                          verbose=True)

Pool2dParams = namedtuple('Pool2dParams', ['stride', 'ksize'], verbose=True)

FullyConLayersParams = namedtuple('FullyConLayersParams',
                                  ['sizes', 'mean', 'stddev', 'rl', 'act_fn'],
                                  verbose=True)

Conv2dLayersParams = namedtuple('Conv2dLayersParams',
                                ['layers', 'mean', 'stddev', 'rl', 'act_fn'],
                                verbose=True)

ModelParams = namedtuple('ModelParams', ['conv_params', 'fc_params', 'pool_params'],
                         verbose=True)


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


def inference(images, model_params):
    """
        Builds cifar10 model.

        Parameters:
        -------
            images: input data
            model_params: ModelParams objects, describes all layers

        Returns:
        -------
            Logits for each label
    """
    fc_params = model_params.fc_params
    pool_params = model_params.pool_params
    conv_params = model_params.conv_params

    with tf.device('/CPU:0'):
        with slim.arg_scope([slim.conv2d], activation_fn=conv_params.act_fn,
                weights_initializer=tf.truncated_normal_initializer
                                (conv_params.mean, conv_params.stddev),
                weights_regularizer=slim.l2_regularizer(conv_params.rl)):
            net = slim.conv2d(images, conv_params.layers[0].filters_count,
                              kernel_size=conv_params.layers[0].ksize,
                              stride=conv_params.layers[0].stride, scope='conv1')
            _tensor_summary(net)
            net = slim.max_pool2d(net, kernel_size=pool_params[0].ksize,
                                  stride=pool_params[0].stride, scope='pool1')
            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm1')


            net = slim.conv2d(net, conv_params.layers[1].filters_count,
                              kernel_size=conv_params.layers[1].ksize,
                              stride=conv_params.layers[1].stride, scope='conv2')
            _tensor_summary(net)
            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                            name='norm2')
            net = slim.max_pool2d(net, kernel_size=pool_params[1].ksize,
                                  stride=pool_params[1].stride, scope='pool2')

            net = tf.reshape(net, [images.get_shape()[0].value, -1])



        with slim.arg_scope([slim.fully_connected], activation_fn=fc_params.act_fn,
                weights_initializer=tf.truncated_normal_initializer
                                    (fc_params.mean, fc_params.stddev),
                weights_regularizer=slim.l2_regularizer(fc_params.rl)):
            net = slim.stack(net, slim.fully_connected, fc_params.sizes[0:2],
                             scope='fc')


        net = slim.fully_connected(net, fc_params.sizes[2], activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer
                        (0, 1 / fc_params.sizes[1]),
            weights_regularizer=slim.l2_regularizer(0.0), scope='softmax-linear')
        _tensor_summary(net)

    return net

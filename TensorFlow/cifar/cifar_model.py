"""
    Cifar10 Network model.
"""

# pylint: disable=C0103
# pylint: disable=C0330

from collections import namedtuple

import tensorflow as tf
slim = tf.contrib.slim

ModelParams = namedtuple('ModelParams', ['filters_count',
                                         'conv_ksizes', 'conv_strides',
                                         'pool_ksizes', 'pool_strides',
                                         'fc_sizes', 'dropouts'],
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
    filters_count = model_params.filters_count
    conv_ksizes = model_params.conv_ksizes
    conv_strides = model_params.conv_strides
    pool_ksizes = model_params.pool_ksizes
    pool_strides = model_params.pool_strides
    fc_sizes = model_params.fc_sizes
    dropouts = model_params.dropouts

    if not filters_count:
        raise ValueError("List of convolutional layers filters is empty!")
    if not conv_ksizes:
        raise ValueError("List of convolutional layers kernel sizes is empty!")
    if not pool_ksizes:
        raise ValueError("List of pool layers kernel sizes is empty!")
    if not fc_sizes:
        raise ValueError("List of full connected layers sizes is empty!")

    conv_layers_count = len(filters_count)
    if len(conv_ksizes) < conv_layers_count:
        conv_ksizes.extend([conv_ksizes[-1]] * (conv_layers_count - len(conv_ksizes)))

    if not conv_strides:
        conv_strides = [1] * conv_layers_count
    elif len(conv_strides) < conv_layers_count:
        conv_strides.extend([conv_strides[-1]] * (conv_layers_count - len(conv_strides)))

    if len(pool_ksizes) < conv_layers_count:
        pool_ksizes.extend([pool_ksizes[-1]] * (conv_layers_count - len(pool_ksizes)))

    if not pool_strides:
        pool_strides = pool_ksizes
    elif len(pool_strides) < conv_layers_count:
        pool_strides.extend([pool_strides[-1]] * (conv_layers_count - len(pool_strides)))

    if not dropouts:
        dropouts = [1] * (conv_layers_count + len(fc_sizes))
    elif len(dropouts) < conv_layers_count:
        dropouts.extend([1] * (conv_layers_count + len(fc_sizes) - len(dropouts) - 1))


    return 5

    # with tf.device('/CPU:0'):
    #     with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
    #             weights_initializer=slim.xavier_initializer()):
    #         net = slim.conv2d(images, conv_params.layers[0].filters_count,
    #                           kernel_size=conv_params.layers[0].ksize,
    #                           stride=conv_params.layers[0].stride, scope='conv1')
    #         _tensor_summary(net)
    #         net = slim.max_pool2d(net, kernel_size=pool_params[0].ksize,
    #                               stride=pool_params[0].stride, scope='pool1')
    #         net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                         name='norm1')


    #         net = slim.conv2d(net, conv_params.layers[1].filters_count,
    #                           kernel_size=conv_params.layers[1].ksize,
    #                           stride=conv_params.layers[1].stride, scope='conv2')
    #         _tensor_summary(net)
    #         net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                         name='norm2')
    #         net = slim.max_pool2d(net, kernel_size=pool_params[1].ksize,
    #                               stride=pool_params[1].stride, scope='pool2')

    #         net = tf.reshape(net, [images.get_shape()[0].value, -1])



    #     with slim.arg_scope([slim.fully_connected], activation_fn=fc_params.act_fn,
    #             weights_initializer=slim.xavier_initializer()):
    #         net = slim.stack(net, slim.fully_connected, fc_params.sizes[0:2],
    #                          scope='fc')


    #     net = slim.fully_connected(net, fc_params.sizes[2], activation_fn=None,
    #         weights_initializer=slim.xavier_initializer(), scope='softmax-linear')
    #     _tensor_summary(net)

    # return net

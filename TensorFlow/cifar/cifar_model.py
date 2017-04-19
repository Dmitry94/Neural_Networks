"""
    Cifar10 Network model.
"""

from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import slim

total_params_count = 0
ModelParams = namedtuple("ModelParams", ["filters_counts",
                                         "conv_ksizes", "conv_strides",
                                         "pool_ksizes", "pool_strides",
                                         "fc_sizes", "drop_rates",
                                         "data_format"])


def _tensor_summary(tensor):
    """
        Generates summury for some tensor.
        Summary it"s histogram and sparsity.
    """
    tensor_name = tensor.op.name
    tf.summary.histogram(tensor_name, tensor)

    # Tensor sparsiry, i.e. zeros_count / all
    tf.summary.scalar(tensor_name + "/sparsity",
                      tf.nn.zero_fraction(tensor))


def conv_pool_drop_2d(in_data, filters_count, conv_ksize, conv_stride,
                      pool_ksize, pool_stride, drop_rate, data_format, scope):
    """
        Creating three layers: conv, max_pool, dropout.

        Parameters:
        -------
            in_data: input tensor of data
            conv_ksize: conv kernel size
            conv_stride: conv stride
            pool_ksize: pool kernel size
            pool_stride: pool stride
            drop_rate: probability of that neuron is active
            data_format: NCHW or NHWC
            scope: scope name

        Returns:
        -------
            Out after conv, max pool and drop.
    """
    out = tf.layers.conv2d(in_data, filters_count, kernel_size=conv_ksize,
                           strides=conv_stride,
                           data_format=data_format, name=scope + "/conv")
    _tensor_summary(out)
    if not isinstance(conv_ksize, list):
        conv_ksize = [conv_ksize, conv_ksize]
    if len(conv_ksize) < 2:
        conv_ksize = [conv_ksize[0], conv_ksize[0]]
    params_count = filters_count * (conv_ksize[0] * conv_ksize[1] * 3 + 1)
    print("Convolutional layer: shape = ", out.get_shape(),
          " kernel = ", conv_ksize, " strides = ", conv_stride,
          " filters count = ", filters_count,
          " layer num of params = ", params_count)
    global total_params_count
    total_params_count += params_count

    out = tf.layers.max_pooling2d(out, pool_size=pool_ksize,
                                  strides=pool_stride,
                                  data_format=data_format,
                                  name=scope + "/pool")
    _tensor_summary(out)
    print("Pool layer: shape = ", out.get_shape(),
          ", kernel = ", pool_ksize, ", strides = ", pool_stride)

    if drop_rate != 0:
        out = tf.layers.dropout(out, rate=drop_rate, name=scope + "/dropout")
        print("Dropout layer: drop rate = ", drop_rate)
        _tensor_summary(out)

    return out


def fc_drop(in_data, fc_size, drop_rate, scope):
    """
        Creates two layers: full connected and dropout.

        Parameters:
        -------
            in_data: input tensor of data
            fc_size: size of full connected layer
            drop_rate: probability of that neuron is active
            scope: scope name

        Returns:
        -------
            Out after fc and dropout.
    """
    out = slim.fully_connected(in_data, fc_size, scope=scope + "fc")
    _tensor_summary(out)
    params_count = fc_size * in_data.get_shape()[-1].value
    print("Fully connected layer: shape = ", out.get_shape(),
          " size = ", fc_size,
          " layer num of params = ", params_count)
    global total_params_count
    total_params_count += params_count

    if drop_rate != 0:
        out = tf.layers.dropout(out, rate=drop_rate, name=scope + "/dropout")
        _tensor_summary(out)
        print("Dropout layer: drop rate = ", drop_rate)

    return out


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
    filters_counts = model_params.filters_counts
    conv_ksizes = model_params.conv_ksizes
    conv_strides = model_params.conv_strides
    pool_ksizes = model_params.pool_ksizes
    pool_strides = model_params.pool_strides
    fc_sizes = model_params.fc_sizes
    drop_rates = model_params.drop_rates
    data_formats = [model_params.data_format] * len(filters_counts)
    global total_params_count
    total_params_count = 0

    if not filters_counts:
        raise ValueError("List of convolutional layers filters is empty!")
    if not conv_ksizes:
        raise ValueError("List of convolutional layers kernel sizes is empty!")
    if not pool_ksizes:
        raise ValueError("List of pool layers kernel sizes is empty!")
    if not fc_sizes:
        raise ValueError("List of full connected layers sizes is empty!")

    conv_layers_count = len(filters_counts)
    if len(conv_ksizes) < conv_layers_count:
        conv_ksizes.extend([conv_ksizes[-1]] *
                           (conv_layers_count - len(conv_ksizes)))

    if not conv_strides:
        conv_strides = [1] * conv_layers_count
    elif len(conv_strides) < conv_layers_count:
        conv_strides.extend([conv_strides[-1]] *
                            (conv_layers_count - len(conv_strides)))

    if len(pool_ksizes) < conv_layers_count:
        pool_ksizes.extend([pool_ksizes[-1]] *
                           (conv_layers_count - len(pool_ksizes)))

    if not pool_strides:
        pool_strides = pool_ksizes
    elif len(pool_strides) < conv_layers_count:
        pool_strides.extend([pool_strides[-1]] *
                            (conv_layers_count - len(pool_strides)))

    dropouts_count = conv_layers_count + len(fc_sizes) - 1
    if len(drop_rates) < dropouts_count:
        drop_rates.extend([0] * (dropouts_count - len(drop_rates)))

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=slim.xavier_initializer()):
        net = slim.stack(images, conv_pool_drop_2d, zip(
            filters_counts, conv_ksizes, conv_strides, pool_ksizes,
            pool_strides, drop_rates[0:conv_layers_count], data_formats),
                            scope="conv_layers")

        net = tf.reshape(net, [images.get_shape()[0].value, -1])
        net = slim.stack(net, fc_drop, zip(fc_sizes[:len(fc_sizes) - 1],
                                           drop_rates[conv_layers_count:]),
                         scope="fc_layers")

        logits = slim.fully_connected(net, fc_sizes[-1], activation_fn=None,
                                      scope="logits")
        _tensor_summary(logits)
        params_count = fc_sizes[-1] * net.get_shape()[-1].value
        print("Fully connected layer: shape = ", logits.get_shape(),
              " size = ", fc_sizes[-1],
              " layer num of params = ", params_count)
        total_params_count += params_count
        print("Total params count = ", total_params_count)

    return logits

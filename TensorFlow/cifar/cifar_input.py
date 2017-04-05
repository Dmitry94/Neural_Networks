"""
    Module for reading cifar data.
    Uses tensorflow tools for effective reading.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """
        Read one sample from queue.

        Parameters:
        -------
        filename_queue : queue of strings

        Returns:
        -------
        Structure with fields:
            height, width, depth, label, image(uint8),
            key - record number & filename
    """
    class Cifar10Record(object):
        pass

    result = Cifar10Record()
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.width * result.height * result.depth
    label_bytes = 1
    record_bytes = label_bytes + image_bytes

    # File format: label:r1:r2:...:rn:g1:g2:...:gn:b1:b2:...:bn
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, raw_data = reader.read(filename_queue)
    record_bytes = tf.decode_raw(raw_data, tf.uint8)

    # Get label from raw data
    # Slice takes from begin_index to end_index
    result.label = tf.to_int32(tf.slice(record_bytes, [0], [label_bytes]))

    raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
    result.image = tf.reshape(raw_image, [result.depth, result.height, result.width])
    result.image = tf.transpose(result.image, [1, 2, 0])

    return result
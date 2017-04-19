"""
    Module for reading cifar data.
    Uses tensorflow tools for effective reading.
"""

# pylint: disable=C0103
# pylint: disable=C0330
# pylint: disable=C0111
# pylint: disable=W0201

import os
import tensorflow as tf

import multiprocessing
import threading
import numpy as np


class Cifar10DataManager(object):
    """
        Class for cifar10 data managment.
    """
    NUM_CLASSES = 10
    TRAIN_SIZE = 50000
    TEST_SIZE = 10000
    IM_SIZE = 24

    def __init__(self, batch_size, data, labels,
                 coord, data_format, queue_size=32):
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.i = 0
        self.lock = threading.Lock()
        self.batches_count = data.shape[0] / batch_size
        self.data_format = data_format

        # Init queue parameters
        self.images_pl = tf.placeholder(tf.float32, [
            batch_size, Cifar10DataManager.IM_SIZE,
            Cifar10DataManager.IM_SIZE, 3])
        self.labels_pl = tf.placeholder(tf.int32, [batch_size])
        if self.data_format == 'NCHW':
            self.images_pl = tf.transpose(self.images_pl, [0, 3, 1, 2])
        self.queue = tf.FIFOQueue(queue_size,
                                  [self.images_pl.dtype, self.labels_pl.dtype],
                                  [self.images_pl.get_shape(),
                                   self.labels_pl.get_shape()])
        self.threads = []
        self.coord = coord
        self.enqueue_op = self.queue.enqueue([self.images_pl, self.labels_pl])

    def next_batch(self):
        """
            Return next batch. Cyclic.
        """
        selection = np.s_[self.i * self.batch_size:
                          (self.i + 1) * self.batch_size]
        with self.lock:
            self.i = (self.i + 1) % self.batches_count

        margin = (32 - Cifar10DataManager.IM_SIZE) / 2
        data_batch = self.data[selection, margin:32 - margin,
                               margin:32 - margin]
        labels_batch = self.labels[selection]

        data_batch = data_batch.astype(np.float32)
        data_batch -= np.mean(data_batch)
        data_batch /= 255.0
        labels_batch = labels_batch.astype(np.int32)

        if self.data_format == 'NCHW':
            data_batch = np.transpose(data_batch, axes=[0, 3, 1, 2])
        return data_batch, labels_batch

    def size(self):
        return self.queue.size()

    def dequeue(self):
        output = self.queue.dequeue()
        return output

    def thread_main(self, session):
        while not self.coord.should_stop():
            data, labels = self.next_batch()

            try:
                session.run(self.enqueue_op,
                            feed_dict={self.images_pl: data,
                                       self.labels_pl: labels})
            except tf.errors.CancelledError:
                return

    def start_threads(self, session, n_threads=multiprocessing.cpu_count()):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(session,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads


def train_inputs(data_dir, batch_size, image_size=Cifar10DataManager.IM_SIZE):
    """
        Get train inputs
    """
    return get_cifar10_input(data_dir, batch_size,
                             image_size, False)


def test_inputs(data_dir, batch_size, image_size=Cifar10DataManager.IM_SIZE):
    """
        Get test inputs
    """
    return get_cifar10_input(data_dir, batch_size,
                             image_size, True)


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
    record = tf.decode_raw(raw_data, tf.uint8)

    # Get label from raw data
    # Slice takes from begin_index to end_index
    result.label = tf.cast(tf.slice(record, [0], [label_bytes]), tf.int32)
    raw_image = tf.slice(record, [label_bytes], [image_bytes])
    result.image = tf.reshape(raw_image,
                              [result.depth, result.height, result.width])
    result.image = tf.transpose(result.image, [1, 2, 0])

    return result


def _generate_batch(image, label, min_queue_size,
                    batch_size, is_shuffle):
    """
        Generates batch.

        Parameters:
        -------
            image: 3d tensor [h, w, 3] of float32
            label: 1d tensor of int32
            min_queue_size: min count of samples
            is_shuffle: flag, which tells shuffle batch or not

        Returns:
        -------
            images: 4d tensor [batch_size, h, w, 3] of float32
            labels: 1d tensor [batch_size] of int32
    """
    num_of_threads = 16
    if is_shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size,
            capacity=min_queue_size + 3 * batch_size,
            min_after_dequeue=min_queue_size, num_threads=num_of_threads)
    else:
        images, labels = tf.train.batch(
            [image, label], batch_size, num_threads=num_of_threads,
            capacity=min_queue_size + 3 * batch_size)

    tf.summary.image('Images', images)
    labels = tf.reshape(labels, [batch_size])

    return images, labels


def get_cifar10_input(data_dir, batch_size, image_size, is_test):
    """
        Constructs input from cifar10.

        Parameters:
        -------
            data_dir: directory where cifar data is allocated
            batch_size: size of one batch
            is_test: flag, defines with *.bins to load

        Returns:
        -------
            images: 4d tensor: [batch_size, h, w, 3]
            labels: 1d tensor: [batch_size]
    """
    if is_test:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = Cifar10DataManager.TEST_SIZE
    else:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = Cifar10DataManager.TRAIN_SIZE

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    queue = tf.train.string_input_producer(filenames)
    records = read_cifar10(queue)
    float_image = tf.cast(records.image, tf.float32)
    float_image = tf.image.resize_image_with_crop_or_pad(float_image,
                                                         image_size,
                                                         image_size)
    float_image = tf.image.per_image_standardization(float_image)

    min_percent_samples_in_queue = 0.4
    min_queue_size = int(min_percent_samples_in_queue * num_examples_per_epoch)

    return _generate_batch(float_image, records.label,
                           min_queue_size, batch_size, False)

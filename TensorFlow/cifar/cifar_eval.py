"""
    Cifar evaluation on test script.
"""

# pylint: disable=C0103
# pylint: disable=C0330

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import math
import os
import time
import h5py
import datetime
import numpy as np

import cifar_model
import cifar_input
import tensorflow as tf


def eval_once(app_args, saver, summary_writer, summary_op):
    """
        Run Eval once.
        Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    train_hdf5 = h5py.File(app_args.data_file, "r")
    coord = tf.train.Coordinator()
    manager = cifar_input.Cifar10DataManager(
        app_args.batch_size, train_hdf5["data"], train_hdf5["labels"],
        coord, app_args.data_format)
    with tf.device('/CPU:0'):
        images, labels = manager.dequeue()

    logits = tf.get_collection('logits')[0]
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(app_args.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = tf.train.get_global_step()
        else:
            print('No checkpoint in this directory')
            return

        threads = manager.start_threads(sess)
        num_iter = int(math.ceil(app_args.samples_count / app_args.batch_size))
        true_count = 0
        total_sample_count = num_iter * app_args.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run([top_k_op], feed_dict={'images:0': images,
                                                          'labels:0': labels})
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision', simple_value=precision)
        summary_writer.add_summary(summary, global_step)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(app_args):
    graph_path = ""
    for file in os.listdir(app_args.checkpoint_dir):
        if file.endswith(".meta"):
            if file > graph_path:
                graph_path = file
    graph_path = os.path.join(app_args.checkpoint_dir, graph_path)

    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(graph_path)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(app_args.log_dir, graph)

        while True:
            eval_once(app_args, saver, summary_writer, summary_op)
            if app_args.eval_once:
                break
            else:
                time.sleep(app_args.eval_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file',
                        help='Path to the data directory',
                        default='../../content/ciraf/hdf5/train.hdf5')

    parser.add_argument('--log-dir',
                        help='Path to the directory, where log will write',
                        default='cifar10_eval')

    parser.add_argument('--checkpoint-dir',
                        help='Path to the directory, where checkpoint stores',
                        default='cifar10_train')

    parser.add_argument('--batch-size', type=int,
                        help='Number of images to process in a batch',
                        default=128)

    parser.add_argument('--samples-count', type=int,
                        help='Number of images to process at all',
                        default=128)

    parser.add_argument('--eval-interval', type=int,
                        help='How often to evaluate',
                        default=60 * 10)

    parser.add_argument('--data-format',
                        help="Data format: NCHW or NHWC",
                        default='NHWC')

    app_args = parser.parse_args()
    if tf.gfile.Exists(app_args.log_dir):
        tf.gfile.DeleteRecursively(app_args.log_dir)
    tf.gfile.MakeDirs(app_args.log_dir)
    tf.logging.set_verbosity('DEBUG')
    evaluate(app_args)

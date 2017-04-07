"""
    Cifar evaluation on test script.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from datetime import datetime
import math
import time

import tensorflow as tf
import numpy as np
import cifar_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'cifar10_eval',
                           "Directory for summary")
tf.app.flags.DEFINE_string('check_pnt_dir', 'cifar10_train',
                           "Directory, where checkpoint stores")
tf.app.flags.DEFINE_integer('num_samples', 10000,
                            "Count of samples from test")
tf.app.flags.DEFINE_integer('eval_interval', 60 * 5,
                            "Evaluating interval in seconds")
tf.app.flags.DEFINE_bool('eval_once', False,
                         "Evaluate once or with some interval?")

def eval_once(saver, summary_writer, top_k_op, summary_op):
    """
        Eval on test once.

        Parameters:
        -------
        saver: saver
        summary_writer: summary writer
        top_k_op: top K op
        summary_op: summary op
    """
    with tf.Session() as session:
        # Restore checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.check_pnt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint in this directory')
            return

        coord = tf.train.Coordinator()
        try:
            # Start queue runners
            threads = tf.train.start_queue_runners(session, coord)

            num_iter = int(math.ceil(FLAGS.num_samples / FLAGS.batch_size))
            true_count = 0
            samples_count = num_iter * FLAGS.batch_size
            step = 0

            while step < num_iter and not coord.should_stop():
                predictions = session.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            accuracy = true_count / samples_count
            print('%s: Accuracy @ 1 = %.3f' % (datetime.now(), accuracy))

            summary = tf.Summary()
            summary.ParseFromString(session.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=accuracy)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    """
        Evaluate net on CIFAR10 Test set.
    """
    with tf.Graph().as_default() as graph:
        images, labels = cifar_model.test_inputs()
        logits = cifar_model.inference(images)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_avgs = tf.train.ExponentialMovingAverage(cifar_model.MOV_AVG_DECAY)
        variables_to_restore = variable_avgs.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.test_dir, graph)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.eval_once:
                break
            else:
                time.sleep(FLAGS.eval_interval)

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()

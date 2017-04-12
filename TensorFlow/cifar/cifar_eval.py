"""
    Cifar evaluation on test script.
"""

# pylint: disable=C0103
# pylint: disable=C0330

import math

import cifar_model
import cifar_input

import tensorflow as tf
slim = tf.contrib.slim


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../../content/ciraf/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_string('test_dir', 'cifar10_eval',
                           "Directory for summary")

tf.app.flags.DEFINE_string('check_pnt_dir', 'cifar10_train',
                           "Directory, where checkpoint stores")


tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")


tf.app.flags.DEFINE_integer('eval_interval', 60,
                            "Evaluating interval in seconds")

tf.app.flags.DEFINE_bool('eval_once', None,
                         "Evaluate once or with some interval?")


def evaluate():
    """
        Evaluate net on CIFAR10 Test set.
    """
    images, labels = cifar_input.test_inputs(FLAGS.data_dir, FLAGS.batch_size)
    logits = cifar_model.inference(images, cifar_input.NUM_CLASSES)
    logits = tf.argmax(logits, axis=1)
    logits = tf.cast(logits, tf.int32)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'accuracy': slim.metrics.accuracy(logits, labels)
    })
    summary_ops = []
    for metric_name, metric_value in names_to_values.iteritems():
        op = tf.summary.scalar(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    num_iter = int(math.ceil(cifar_input.TEST_SIZE / FLAGS.batch_size))

    slim.evaluation.evaluation_loop(
        'local',
        FLAGS.check_pnt_dir,
        FLAGS.test_dir,
        num_evals=num_iter,
        eval_op=names_to_updates.values(),
        summary_op=tf.summary.merge(summary_ops),
        max_number_of_evaluations=FLAGS.eval_once,
        eval_interval_secs=FLAGS.eval_interval)



if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)
    evaluate()

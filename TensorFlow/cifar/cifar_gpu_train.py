"""
    Train cifar model on GPUs.
"""

# pylint: disable=C0103
# pylint: disable=C0330

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import tensorflow as tf
import cifar_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Max steps for algo')
tf.app.flags.DEFINE_integer('gpus_count', 1, 'Count of GPUs')
tf.app.flags.DEFINE_string('train_dir', 'cifar10_gpu_train', 'Directory for train log')
tf.app.flags.DEFINE_boolean('log_dev_placement', False, 'Add to log device placement?')

TOWER_NAME = 'TOWER'

def tower_loss(scope):
    """
        Calculates tower loss, usint it's scope.

        Parameters:
        -------
        scope: string - name of the scope.

        Returns:
        -------
        Total loss value for tower.
    """
    images, labels = cifar_model.train_inputs()
    logits = cifar_model.inference(images)

    _ = cifar_model.loss(labels, logits)

    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, 'total_loss')

    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss

def average_gradients(grads):
    """
        Calculates average gradient between towers.
        Synch point for all towers.

        Parameters:
        -------
        grads: List of lists of (gradient, variable) tuples.

        Returns:
        List of (gradient, variable)
    """
    average_grads = []

    # [[(gr, v)]] -> [((gr, v), ..., (gr_n, v_n))]
    for grad_and_vars in zip(*grads):
        grads = []
        for gr, _ in grad_and_vars:
            expanded_gr = tf.expand_dims(gr, 0)
            grads.append(expanded_gr)

        grads = tf.concat(grads, axis=0)
        grads = tf.reduce_mean(grads, axis=0)

        # We should return grads and variables in one tuple
        v = grad_and_vars[0][1]
        avg_gr_and_vars = (grads, v)
        average_grads.append(avg_gr_and_vars)

    return average_grads

def train():
    """
        Train model.
    """
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = (cifar_model.TRAIN_SIZE /
                                    FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * cifar_model.NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(cifar_model.INIT_LR,
                                        global_step,
                                        decay_steps,
                                        cifar_model.LR_DECAY_FACTOR,
                                        staircase=True)

        opt = tf.train.GradientDescentOptimizer(lr)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.gpus_count):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        loss = tower_loss(scope)
                        tf.get_variable_scope().reuse_variables()

                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grad = opt.compute_gradients(loss)
                        tower_grads.append(grad)

        grads = average_gradients(tower_grads)
        summaries.append(tf.summary.scalar('Learning rate', lr))

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        var_mov_avgs = tf.train.ExponentialMovingAverage(cifar_model.MOV_AVG_DECAY,
                                global_step)
        var_mov_avgs_op = var_mov_avgs.apply(tf.trainable_variables())


        train_op = tf.group(apply_gradient_op, var_mov_avgs_op)
        saver = tf.train.Saver(tf.trainable_variables())
        summary_op = tf.summary.merge(summaries)

        init = tf.global_variables_initializer()


        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                    log_device_placement=FLAGS.log_dev_placement))
        session.run(init)

        tf.train.start_queue_runners(session)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, session.graph)


        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = session.run([train_op, loss])
            during = time.time() - start_time

            if step % 10 == 0:
                exmpls_per_step = FLAGS.batch_size * FLAGS.gpus_count
                exmpls_per_sec = exmpls_per_step / during
                scds_per_batch = during / FLAGS.gpus_count

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                    exmpls_per_sec, scds_per_batch))

            if step % 100 == 0:
                summary_str = session.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'cifar.ckpt')
                saver.save(session, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
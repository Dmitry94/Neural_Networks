"""
  Train cifar model.
"""

import os
import argparse
import time
import h5py
import numpy as np

import cifar_model
import cifar_input

import tensorflow as tf
from tensorflow.contrib import slim

train_hdf5 = h5py.File("../../content/ciraf/hdf5/train.hdf5", "r")


class Cifar10DataManager(object):
    """
        Class for cifar10 data managment.
    """
    def __init__(self, batch_size, data, labels):
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.i = 0
        self.batches_count = data.shape[0] / batch_size

        self.data_batch = np.zeros((batch_size,
                                    cifar_input.IM_SIZE,
                                    cifar_input.IM_SIZE,
                                    self.data.shape[3]))
        self.labels_batch = np.zeros((batch_size,))

    def next_batch(self):
        """
            Return next batch. Cyclic.
        """
        selection = np.s_[self.i * self.batch_size:
                          (self.i + 1) * self.batch_size]
        self.i = (self.i + 1) % self.batches_count

        margin = (32 - cifar_input.IM_SIZE) / 2
        data_batch = self.data[selection, margin:32-margin, margin:32-margin]
        labels_batch = self.labels[selection]

        data_batch = data_batch.astype(np.float32)
        data_batch -= np.mean(data_batch)
        data_batch /= 255.0
        labels_batch = labels_batch.astype(np.int32)

        return data_batch, labels_batch

    def next_batch_direct(self):
        """
            Return next batch. Cyclic.
        """
        margin = (32 - cifar_input.IM_SIZE) / 2
        data_sel = np.s_[self.i * self.batch_size:
                         (self.i + 1) * self.batch_size,
                         margin:32-margin, margin:32-margin]
        labels_sel = np.s_[self.i * self.batch_size:
                           (self.i + 1) * self.batch_size]
        self.i = (self.i + 1) % self.batches_count

        self.data.read_direct(self.data_batch, data_sel, np.s_[:])
        self.labels.read_direct(self.labels_batch, labels_sel, np.s_[:])
        self.data_batch = self.data_batch.astype(np.float32)
        self.data_batch -= np.mean(self.data_batch)
        self.data_batch /= 255.0
        self.labels_batch = self.labels_batch.astype(np.int32)

        return self.data_batch, self.labels_batch


def get_model_params(app_args):
    """
        Creating ModelParams object.
    """
    model_params = cifar_model.ModelParams(
        filters_counts=app_args.filters_counts,
        conv_ksizes=app_args.conv_ksizes,
        conv_strides=app_args.conv_strides,
        pool_ksizes=app_args.pool_ksizes,
        pool_strides=app_args.pool_strides,
        fc_sizes=app_args.fc_sizes,
        drop_rates=app_args.drop_rates)

    return model_params


def train(app_args):
    """
      Train CIFAR-10 for a number of steps.
    """
    manager = Cifar10DataManager(app_args.batch_size, train_hdf5["data"],
                                 train_hdf5["labels"])
    with tf.Graph().as_default() as graph:
        # Get images and labels for CIFAR-10.
        images = tf.placeholder(tf.float32, [app_args.batch_size,
                                             cifar_input.IM_SIZE,
                                             cifar_input.IM_SIZE, 3])
        labels = tf.placeholder(tf.int32, [app_args.batch_size])

        # Build a Graph that computes the logits predictions
        model_params = get_model_params(app_args)
        logits = cifar_model.inference(images, model_params)

        # Calculate loss.
        tf.losses.sparse_softmax_cross_entropy(labels, logits)
        loss = tf.losses.get_total_loss()

        # Set learning rate and optimizer
        global_step = tf.contrib.framework.get_or_create_global_step()
        num_batches_per_epoch = cifar_input.TRAIN_SIZE / app_args.batch_size
        lr_decay_steps = app_args.num_epochs_lr_decay * num_batches_per_epoch
        lr = tf.train.exponential_decay(app_args.init_lr,
                                        global_step,
                                        lr_decay_steps,
                                        app_args.lr_decay_factor,
                                        staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)

        # Define ops
        init_op = tf.global_variables_initializer()
        train_op = slim.learning.create_train_op(loss, opt)

        tf.summary.scalar('Learning_rate', lr)
        tf.summary.scalar('Loss', loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(app_args.log_dir, graph)

        saver = tf.train.Saver()

        with tf.Session() as session:
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator)

            session.run(init_op)
            start_time = time.time()

            for step in xrange(app_args.max_steps):
                images_feed, labels_feed = manager.next_batch()
                feed_dict = {
                    images: images_feed,
                    labels: labels_feed
                }
                _, loss_value = session.run([train_op, loss],
                                            feed_dict=feed_dict)

                if step % app_args.save_summary_steps == 0 and step > 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary = session.run(summary_op, feed_dict=feed_dict,
                                          options=run_options,
                                          run_metadata=run_metadata)

                    summary_writer.add_run_metadata(run_metadata,
                                                    'step%d' % step)
                    summary_writer.add_summary(summary, step)

                    current_time = time.time()
                    duration = current_time - start_time
                    start_time = current_time

                    examples_per_sec = int(app_args.save_summary_steps *
                                           app_args.batch_size / duration)
                    sec_per_batch = float(duration /
                                          app_args.save_summary_steps)
                    print(
                        'Step = %d Loss = %f Samples per sec = %d'
                        ' Sec per batch = %f' %
                        (step, loss_value, examples_per_sec, sec_per_batch))

                if step % app_args.save_checkpoint_steps == 0:
                    checkpoint_file = os.path.join(app_args.log_dir,
                                                   'model.ckpt')
                    saver.save(session, checkpoint_file, step)

            coordinator.request_stop()
            coordinator.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        help='Path to the data directory',
                        default='../../content/ciraf/cifar-10-batches-bin')

    parser.add_argument('--log-dir',
                        help='Path to the directory, where log will write',
                        default='cifar10_train')

    parser.add_argument('--max-steps', type=int,
                        help='Number of batches to run',
                        default=1000000)

    parser.add_argument('--batch-size', type=int,
                        help='Number of images to process in a batch',
                        default=128)

    parser.add_argument('--init-lr', type=float,
                        help='Start value for learning rate',
                        default=0.1)

    parser.add_argument('--lr-decay-factor', type=float,
                        help='Learning rate decay factor',
                        default=0.1)

    parser.add_argument('--num-epochs-lr-decay', type=int,
                        help='How many epochs should processed to decay lr',
                        default=350)

    parser.add_argument('--log-frequency', type=int,
                        help='How often to log results to the console',
                        default=10)

    parser.add_argument('--save-checkpoint-steps', type=int,
                        help='How often to save checkpoint',
                        default=1000)

    parser.add_argument('--save-summary-steps', type=int,
                        help='How often to save summary',
                        default=100)

    parser.add_argument('--filters-counts', nargs='+', type=int,
                        help='List of filter counts for each conv layer',
                        default=[64, 64])

    parser.add_argument('--conv-ksizes', nargs='+', type=int,
                        help='List of kernel sizes for each conv layer',
                        default=[5])

    parser.add_argument('--conv-strides', nargs='+', type=int,
                        help='List of strides for each conv layer',
                        default=[])

    parser.add_argument('--pool-ksizes', nargs='+', type=int,
                        help='List of kernel sizes for each pool layer',
                        default=[3])

    parser.add_argument('--pool-strides', nargs='+', type=int,
                        help='List of strides for each pool layer',
                        default=[2])

    parser.add_argument('--fc-sizes', nargs='+', type=int,
                        help='List of sizes for each fc layer',
                        default=[384, 192, cifar_input.NUM_CLASSES])

    parser.add_argument('--drop-rates', nargs='+', type=int,
                        help="List of probs for each conv and fc layer",
                        default=[])

    app_args = parser.parse_args()

    if tf.gfile.Exists(app_args.log_dir):
        tf.gfile.DeleteRecursively(app_args.log_dir)
    tf.gfile.MakeDirs(app_args.log_dir)
    tf.logging.set_verbosity('DEBUG')
    train(app_args)

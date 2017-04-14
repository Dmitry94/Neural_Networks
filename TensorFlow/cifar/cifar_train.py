"""
  Train cifar model.
"""

import argparse

import cifar_model
import cifar_input

import tensorflow as tf
from tensorflow.contrib import slim


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
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    with tf.device('/CPU:0'):
        images, labels = cifar_input.train_inputs(app_args.data_dir,
                                                  app_args.batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    model_params = get_model_params(app_args)
    logits = cifar_model.inference(images, model_params)

    # Calculate loss.
    tf.losses.sparse_softmax_cross_entropy(labels, logits)
    loss = tf.losses.get_total_loss()

    # Set learning rate and optimizer
    num_batches_per_epoch = cifar_input.TRAIN_SIZE / app_args.batch_size
    lr_decay_steps = app_args.num_epochs_lr_decay * num_batches_per_epoch
    lr = tf.train.exponential_decay(app_args.init_lr, global_step,
                                    lr_decay_steps, app_args.lr_decay_factor,
                                    staircase=True)
    opt = tf.train.GradientDescentOptimizer(lr)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = slim.learning.create_train_op(loss, opt)

    tf.summary.scalar('Learning_rate', lr)
    tf.summary.scalar('Loss', loss)

    slim.learning.train(train_op, app_args.log_dir,
                        number_of_steps=app_args.max_steps,
                        save_summaries_secs=app_args.save_summary_secs,
                        save_interval_secs=app_args.save_checkpoint_secs,
                        log_every_n_steps=app_args.log_frequency)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='Path to the data directory',
                        default='../../content/ciraf/cifar-10-batches-bin')

    parser.add_argument('--log_dir',
                        help='Path to the directory, where log will write',
                        default='cifar10_train')

    parser.add_argument('--max_steps', type=int,
                        help='Number of batches to run',
                        default=1000000)

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch',
                        default=128)

    parser.add_argument('--init_lr', type=float,
                        help='Start value for learning rate',
                        default=0.1)

    parser.add_argument('--lr_decay_factor', type=float,
                        help='Learning rate decay factor',
                        default=0.1)

    parser.add_argument('--num_epochs_lr_decay', type=int,
                        help='How many epochs should processed to decay lr',
                        default=350)

    parser.add_argument('--log_frequency', type=int,
                        help='How often to log results to the console',
                        default=10)

    parser.add_argument('--save_checkpoint_secs', type=int,
                        help='How often to save checkpoint',
                        default=60 * 10)

    parser.add_argument('--save_summary_secs', type=int,
                        help='How often to save summary',
                        default=60 * 5)

    parser.add_argument('--filters_counts', nargs='+', type=int,
                        help='List of filter counts for each conv layer',
                        default=[64, 64])

    parser.add_argument('--conv_ksizes', nargs='+', type=int,
                        help='List of kernel sizes for each conv layer',
                        default=[5])

    parser.add_argument('--conv_strides', nargs='+', type=int,
                        help='List of strides for each conv layer',
                        default=[])

    parser.add_argument('--pool_ksizes', nargs='+', type=int,
                        help='List of kernel sizes for each pool layer',
                        default=[3])

    parser.add_argument('--pool_strides', nargs='+', type=int,
                        help='List of strides for each pool layer',
                        default=[2])

    parser.add_argument('--fc_sizes', nargs='+', type=int,
                        help='List of sizes for each fc layer',
                        default=[384, 192, cifar_input.NUM_CLASSES])

    parser.add_argument('--drop_rates', nargs='+', type=int,
                        help="List of probs for each conv and fc layer",
                        default=[])

    app_args = parser.parse_args()

    if tf.gfile.Exists(app_args.log_dir):
        tf.gfile.DeleteRecursively(app_args.log_dir)
    tf.gfile.MakeDirs(app_args.log_dir)
    tf.logging.set_verbosity('DEBUG')
    train(app_args)

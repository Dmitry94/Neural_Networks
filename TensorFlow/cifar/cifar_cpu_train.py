"""
  Train cifar model.
"""

import tensorflow as tf
slim = tf.contrib.slim

import cifar_model
import cifar_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../../content/ciraf/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_string('train_dir', 'cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")


tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")


tf.app.flags.DEFINE_float('init_lr', 0.1, """Start value for learning rate""")

tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, """Learning rate decay factor""")

tf.app.flags.DEFINE_integer('num_epochs_lr_decay', 350,
                            """How many epochs should processed to decay lr.""")


tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

tf.app.flags.DEFINE_integer('save_checkpoint_secs', 60 * 10,
                            """How often to save checkpoint.""")

tf.app.flags.DEFINE_integer('save_summary_secs', 60 * 5,
                            """How often to save summary.""")


def train():
    """
      Train CIFAR-10 for a number of steps.
    """
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = cifar_input.train_inputs(FLAGS.data_dir, FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar_model.inference(images, cifar_input.NUM_CLASSES)

        # Calculate loss.
        _ = slim.losses.sparse_softmax_cross_entropy(logits, labels)
        loss = slim.losses.get_total_loss()

        # Set learning rate and optimizer
        num_batches_per_epoch = cifar_input.TRAIN_SIZE / FLAGS.batch_size
        lr_decay_steps = FLAGS.num_epochs_lr_decay * num_batches_per_epoch
        lr = tf.train.exponential_decay(FLAGS.init_lr, global_step, lr_decay_steps,
                                        FLAGS.lr_decay_factor, staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = slim.learning.create_train_op(loss, opt)

        tf.summary.scalar('Learning rate', lr)
        tf.summary.scalar('Loss', loss)

        slim.learning.train(train_op, FLAGS.train_dir,
                            number_of_steps=FLAGS.max_steps,
                            save_summaries_secs=FLAGS.save_summary_secs,
                            save_interval_secs=FLAGS.save_checkpoint_secs,
                            log_every_n_steps=FLAGS.log_frequency)


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

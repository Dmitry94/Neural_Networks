"""
  Train cifar model.
"""

# pylint: disable=C0103
# pylint: disable=C0330

import cifar_model
import cifar_input

import tensorflow as tf
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../../content/ciraf/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_string('train_dir', 'cifar10_train',
                           """Directory where to write event logs and checkpoint.""")


tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")


tf.app.flags.DEFINE_float('init_lr', 0.1,
                          """Start value for learning rate""")

tf.app.flags.DEFINE_float('lr_decay_factor', 0.1,
                          """Learning rate decay factor""")

tf.app.flags.DEFINE_integer('num_epochs_lr_decay', 350,
                            """How many epochs should processed to decay lr.""")


tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

tf.app.flags.DEFINE_integer('save_checkpoint_secs', 60 * 10,
                            """How often to save checkpoint.""")

tf.app.flags.DEFINE_integer('save_summary_secs', 60 * 5,
                            """How often to save summary.""")

def get_model_params():
    """
        Creating ModelParams object.
    """
    conv_layer1 = cifar_model.Conv2dParams(ksize=5, stride=1, filters_count=64)
    conv_layer2 = cifar_model.Conv2dParams(ksize=5, stride=1, filters_count=64)
    conv_params = cifar_model.Conv2dLayersParams(layers=[conv_layer1, conv_layer2],
                                                 mean=0, stddev=5e-2, rl=0.0,
                                                 act_fn=tf.nn.relu)

    pool_layer1 = cifar_model.Pool2dParams(ksize=3, stride=2)
    pool_layer2 = cifar_model.Pool2dParams(ksize=3, stride=2)
    pool_params = [pool_layer1, pool_layer2]

    fc_params = cifar_model.FullyConLayersParams(sizes=[384, 192, cifar_input.NUM_CLASSES],
                                                 mean=0, stddev=4e-2, rl=0.004,
                                                 act_fn=tf.nn.relu)

    model_params = cifar_model.ModelParams(conv_params=conv_params,
                                           pool_params=pool_params,
                                           fc_params=fc_params)

    return model_params

def train():
    """
      Train CIFAR-10 for a number of steps.
    """
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = cifar_input.train_inputs(FLAGS.data_dir,
                                                  FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        model_params = get_model_params()
        logits = cifar_model.inference(images, model_params)

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

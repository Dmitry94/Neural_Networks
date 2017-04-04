"""
    Demonstrating TensorFlow for MNIST solving.
    Programm origanize in right order.
"""

import math
import time
import argparse
import os.path
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE ** 2
FLAGS = None

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def inference(images, hl1, hl2):
    """
        Forward pass for images.
    """
    stddev = 1.0 / math.sqrt(float(IMAGE_PIXELS))

    # First hiddent layer
    with tf.name_scope('first_hidden_layer'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hl1],
                                stddev=stddev),
            name='weights'
        )
        biases = tf.Variable(
            tf.zeros([hl1], dtype=tf.float32),
            name='biases'
        )
        h1_out = tf.nn.relu(tf.matmul(images, weights) + biases)

        variable_summaries(weights)
        variable_summaries(biases)
        tf.summary.histogram('h1_act', h1_out)

    # Second hidden layer
    with tf.name_scope('second_hidden_layer'):
        weights = tf.Variable(
            tf.truncated_normal([hl1, hl2],
                                stddev=stddev),
            name='weights'
        )
        biases = tf.Variable(
            tf.zeros([hl2], dtype=tf.float32),
            name='biases'
        )
        h2_out = tf.nn.relu(tf.matmul(h1_out, weights) + biases)

        variable_summaries(weights)
        variable_summaries(biases)
        tf.summary.histogram('h2_act', h2_out)

    # Last out layer
    with tf.name_scope('softmax'):
        weights = tf.Variable(
            tf.truncated_normal([hl2, NUM_CLASSES],
                                stddev=stddev),
            name='weights'
        )
        biases = tf.Variable(
            tf.zeros([NUM_CLASSES], dtype=tf.float32),
            name='biases'
        )
        logits = tf.matmul(h2_out, weights) + biases

        variable_summaries(weights)
        variable_summaries(biases)
        tf.summary.histogram('logits', logits)

    return logits


def loss(logits, labels):
    """
        Calculate cross-entropy loss.
    """
    labels = tf.to_int64(labels)
    with tf.name_scope('cross_entropy'):
        cr_en = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(cr_en, name='xentropy_mean')
    return cross_entropy


def train(loss_, learning_rate):
    """
        Train model via loss gd.
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('train'):
        train_op = optimizer.minimize(loss_, global_step)

    return train_op

def evaluate(labels, logits):
    """
        Calculates count of true-predicted.
    """
    predictions = tf.nn.softmax(logits)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return tf.reduce_sum(tf.cast(correct_prediction, tf.int32))



def placeholder_in(batch_size):
    """
        Generates placeholders for images and labels.
    """
    images_pl = tf.placeholder(tf.float32, [batch_size, IMAGE_PIXELS])
    labels_pl = tf.placeholder(tf.int32, [batch_size, NUM_CLASSES])

    return images_pl, labels_pl

def fill_dict(data_set, im_pl, lb_pl):
    """
        Fill dictionary.
    """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)

    feed_dict = {
        im_pl: images_feed,
        lb_pl: labels_feed
    }
    return feed_dict

def do_eval(session, eval_tensor, im_pl, lb_pl, dataset):
    """
        Runs one eval against full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = dataset.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_dict(dataset, im_pl, lb_pl)
        true_count += session.run(eval_tensor, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():
    """
        Train
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Graph().as_default():
        im_pl, lb_pl = placeholder_in(FLAGS.batch_size)
        logits = inference(im_pl, FLAGS.hl1, FLAGS.hl2)
        loss_ = loss(logits, lb_pl)
        train_op = train(loss_, FLAGS.lr)
        eval_correct_tensor = evaluate(lb_pl, logits)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # For writing checkpoints
        saver = tf.train.Saver()
        session = tf.Session()

        # To output summaries and the graph
        summ_writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
        session.run(init)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            feed_dict = fill_dict(mnist.train, im_pl, lb_pl)

            # Every tensor return value. train_op -> None
            _, loss_val, s = session.run([train_op, loss_, summary], feed_dict=feed_dict)
            summ_writer.add_summary(s, step)

            duration = time.time() - start_time

            # Every 100th step write log
            if step % 100 == 0:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(session, checkpoint_file, step)
                # Evaluate against the training set.
                print 'Training Data Eval:'
                do_eval(session,
                        eval_correct_tensor,
                        im_pl, lb_pl,
                        mnist.train)
                # Evaluate against the validation set.
                print 'Validation Data Eval:'
                do_eval(session,
                        eval_correct_tensor,
                        im_pl, lb_pl,
                        mnist.validation)
                # Evaluate against the test set.
                print 'Test Data Eval:'
                do_eval(session,
                        eval_correct_tensor,
                        im_pl, lb_pl,
                        mnist.test)

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--lr',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hl1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hl2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='MNIST_data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='mnist_log',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
"""
    Simple test for ciraf_input based on TensorFlow.
"""

# pylint: disable=C0103
# pylint: disable=C0330
# pylint: disable=C0111

import os
import tensorflow as tf
import cifar_input

def read_ciraf_file(file_name):
    """
        Read all data from CIRAF-file.
    """
    import cPickle
    file_desc = open(file_name, 'rb')
    result = cPickle.load(file_desc)
    file_desc.close()
    return result

class Cifar10ReadingTest(tf.test.TestCase):

    def _record(self, label, red, green, blue):
        image_size = 32 * 32
        record = bytes(bytearray([label] + [red] * image_size +
                                 [green] * image_size +
                                 [blue] * image_size))
        expected = [[[red, green, blue]] * 32] * 32

        return record, expected

    def testSimple(self):
        labels = [3, 2, 8]
        records = [self._record(labels[0], 255, 0, 0),
                   self._record(labels[1], 0, 255, 0),
                   self._record(labels[2], 0, 0, 255)]
        expected = [expected for _, expected in records]
        content = b"".join([record for record, _ in records])

        file_name = os.path.join(self.get_temp_dir(),
                                 'cifar_test')
        open(file_name, "wb").write(content)

        filenames = [os.path.join('../../content/ciraf/cifar-10-batches-bin', 'data_batch_%d.bin' % i)
                     for i in xrange(1, 2)]
        queue = tf.train.string_input_producer(filenames)
        records = cifar_input.read_cifar10(queue)
        with self.test_session() as session:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord)

            for i in xrange(200):
                r = session.run([records.label])
                if r[0] < 0 or r[0] > 10:
                    print r

            coord.request_stop()
            coord.join(threads)


            q = tf.FIFOQueue(99, tf.string, shapes=())
            q.enqueue([file_name]).run()
            q.close().run()
            result = cifar_input.read_cifar10(q)

            for i in xrange(3):
                key, label, image = session.run([result.key,
                                                 result.label,
                                                 result.image])
                self.assertEqual("%s:%d" % (file_name, i), tf.compat.as_text(key))
                self.assertEqual(labels[i], label)
                self.assertAllEqual(expected[i], image)

            with self.assertRaises(tf.errors.OutOfRangeError):
                session.run([result.key, result.image])

if __name__ == "__main__":
    tf.test.main()

"""
    Simple test for ciraf_input based on TensorFlow.
"""

import os
import tensorflow as tf
import cifar_input

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

        with self.test_session() as session:
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

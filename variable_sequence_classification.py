# Working example for my blog post at:
# http://danijar.com/variable-sequence-lengths-in-tensorflow/

import functools
import tensorflow as tf


# def lazy_property(function):
#     attribute = '_' + function.__name__

#     @property
#     @functools.wraps(function)
#     def wrapper(self):
#         if not hasattr(self, attribute):
#             setattr(self, attribute, function(self))
#         return getattr(self, attribute)
#     return wrapper


class VariableSequenceClassification(object):

    def __init__(self, data, target, num_hidden=200, num_layers=2):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction = self.prediction()
        self.error = self.error()
        self.optimize = self.optimize()

    # @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # @lazy_property
    def prediction(self):
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(self._num_hidden),
            self.data,
            dtype=tf.float32,
            sequence_length=self.length(),
        )
        last = self._last_relevant(output, self.length())
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1])
        )
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

        return prediction

    # @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    # @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        return optimizer.minimize(self.cost())

    # @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1)
        )
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets("./mnist/", one_hot=True)
    # from tensorflow.contrib.learn.python.learn.datasets import mnist
    # data = mnist.load_mnist()
    # We treat images as sequences of pixel rows.
    train, valid, test =\
        data.train.images, data.validation.images, data.test.images
    train, valid, test =\
        train.reshape(-1, 28, 28), valid.reshape(-1, 28, 28), test.reshape(-1, 28, 28)
    train_labels, valid_labels, test_labels =\
        data.train.labels, data.validation.labels, data.test.labels
    _, rows, row_size = train.shape
    num_classes = data.train.labels.shape[1]
    data_holder = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    model = VariableSequenceClassification(data_holder, target)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch = 0
        for epoch in range(20):
            for _ in range(100):
                # batch = data.train.next_batch(10)
                sess.run(model.optimize, {data_holder: train[batch:batch+10], target: train_labels[batch:batch+10]})
            error = sess.run(model.error, {data_holder: valid[batch:batch+10], target: valid_labels[batch:batch+10]})
            print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
            batch += 64

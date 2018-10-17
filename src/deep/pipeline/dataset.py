import tensorflow as tf
import numpy as np


class Dataset:
    def __init__(self) -> None:
        self.filename = 'train.tfrecords'

    def write(self, X, Y):
        with tf.python_io.TFRecordWriter(self.filename) as writer:
            for i in range(0, len(X)):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'X': _bytes_feature(X[i].flatten().tostring()),
                            'X_shape': _int64_feature(X[i].shape),
                            'Y': _bytes_feature(Y[i].flatten().tostring()),
                            'Y_shape': _int64_feature(Y[i].shape),
                        }))
                writer.write(example.SerializeToString())

    def read(self):
        dataset = []

        record_iterator = tf.python_io.tf_record_iterator(path=self.filename)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            x_shape = example.features.feature['X_shape'].int64_list.value
            y_shape = example.features.feature['Y_shape'].int64_list.value

            x = example.features.feature['X'].bytes_list.value[0]
            x = np.fromstring(x, dtype=np.float32)
            x = x.reshape(x_shape)

            y = example.features.feature['Y'].bytes_list.value[0]
            y = np.fromstring(y, dtype=np.float32)
            y = y.reshape(y_shape)
            dataset.append((x, y))

        return dataset


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

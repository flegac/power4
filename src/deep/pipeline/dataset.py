import numpy
import numpy as np
import tensorflow as tf


# https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af
class Dataset:

    def __init__(self, filename='my_data.tfrecords') -> None:
        self.filename = filename

    def write(self, X: np.ndarray, Y: np.ndarray):
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
        X = []
        Y = []

        record_iterator = tf.python_io.tf_record_iterator(path=self.filename)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            x = self.read_feature(example, 'X')
            y = self.read_feature(example, 'Y')

            X.append(x)
            Y.append(y)

        X = np.array(X)
        X = np.rollaxis(X, 1, 4)

        Y = np.array(Y)
        return {
            'X': X,
            'Y': Y
        }

    def read_feature(self, example: tf.train.Example, name: str):
        shape = example.features.feature['{}_shape'.format(name)].int64_list.value
        feat = example.features.feature[name].bytes_list.value[0]
        feat = numpy.frombuffer(feat, dtype=np.float32).reshape(shape)
        return feat


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

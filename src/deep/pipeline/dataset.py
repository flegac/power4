import numpy
import numpy as np
import tensorflow as tf


# https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af
class Dataset:

    def __init__(self) -> None:
        self.x = []
        self.y = []

    def set_data(self, x: [np.ndarray], y: [np.ndarray]):
        self.x = x
        self.y = y
        return self

    def write(self, filename: str):
        assert len(self.x) == len(self.y)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for i in range(0, len(self.x)):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'X': _bytes_feature(self.x[i].flatten().tostring()),
                            'X_shape': _int64_feature(self.x[i].shape),
                            'Y': _bytes_feature(self.y[i].flatten().tostring()),
                            'Y_shape': _int64_feature(self.y[i].shape),
                        }))
                writer.write(example.SerializeToString())

    def read(self, filename: str):
        record_iterator = tf.python_io.tf_record_iterator(path=filename)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            _x = self.read_feature(example, 'X')
            _y = self.read_feature(example, 'Y')

            self.x.append(_x)
            self.y.append(_y)
        return self

    @staticmethod
    def read_feature(example: tf.train.Example, name: str):
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

import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self) -> None:
        self.data_set_filename = 'train.tfrecords'

    def write(self, X, Y):
        with tf.python_io.TFRecordWriter(self.data_set_filename) as writer:
            for i in range(0, len(X)):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'X': bytes_feature(X[i]),
                    'Y': bytes_feature(Y[i])
                }))
                writer.write(example.SerializeToString())

    def read(self):
        samples = []

        record_iterator = tf.python_io.tf_record_iterator(path=self.data_set_filename)

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            samples.append({
                'X': self._extract(example, 'X'),
                'Y': self._extract(example, 'Y'),
            })
        return samples

    def _extract(self, example: tf.train.Example, field_name: str):
        field_string = example.features.feature[field_name].bytes_list.value[0]
        return np.fromstring(field_string, dtype=np.uint8)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value)]))

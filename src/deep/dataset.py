import os

import numpy
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af
class Dataset:
    flow_config = {
        'seed': 2210,
        'shuffle': True
        # 'class_mode': 'categorical'
    }

    def __init__(self, name: str, x=None, y=None) -> None:
        self.name = name
        self.x = x or []
        self.y = y or []

    def generator(self, batch_size, **kwargs):
        data_generator = ImageDataGenerator(**kwargs)
        x = np.array(self.x)
        y = np.array(self.y)
        x = np.rollaxis(x, 1, 4)
        return data_generator.flow(x, y,
                                   batch_size=batch_size,
                                   **Dataset.flow_config
                                   )
        # return data_generator.flow_from_directory(self.path,
        #                                           target_size=(height, width),
        #                                           batch_size=batch_size,
        #                                           **Dataset.flow_config)

    def set_data(self, x: [np.ndarray], y: [np.ndarray]):
        self.x = x
        self.y = y
        return self

    def split_by_size(self, size: int):
        split = []
        for i in range(len(self.x) // size):
            split.append(self.extract('{}_' + str(i), size))
        if len(self.x) > 0:
            self.name += '_' + str(len(split))
            split.append(self)
        return split

    def extract(self, name: str, number: int):
        x_extract = self.x[:number]
        y_extract = self.y[:number]
        self.x = self.x[number:]
        self.y = self.y[number:]
        return Dataset(name=name.format(self.name), x=x_extract, y=y_extract)

    def load_all(self, path: str, prefix: str):
        for file in os.listdir(path):
            if file.startswith(prefix):
                print(file)
                self.load(os.path.join(path, file))
        return self

    def filter(self, filter_func):
        x = []
        y = []
        for i in range(len(x)):
            if filter_func(x[i], y[i]):
                x.append(x[i])
                y.append(y[i])
        self.x = x
        self.y = y

    def save(self):
        assert len(self.x) == len(self.y)
        with tf.python_io.TFRecordWriter('{}_n={}.tfrecords'.format(self.name, len(self.x))) as writer:
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

    def load(self, filename: str):
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

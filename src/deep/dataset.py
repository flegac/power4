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

    def __init__(self, name: str,
                 features: set = None,
                 data: dict = None,
                 x_name: str = 'x',
                 y_name: str = 'y') -> None:
        if not data:
            data = {key: [] for key in features}
        if features:
            data = {key: data[key] for key in features}
        self.name = name
        self.data = data
        self.x_name = x_name
        self.y_name = y_name

    def size(self):
        return len(self.get(self.x_name))

    def generator(self, batch_size, **kwargs):
        assert (len(self.get(self.y_name)) == self.size())
        data_generator = ImageDataGenerator(**kwargs)
        x = np.array(self.get(self.x_name))
        y = np.array(self.get(self.y_name))
        x = np.rollaxis(x, 1, 4)
        return data_generator.flow(x, y,
                                   batch_size=batch_size,
                                   **Dataset.flow_config
                                   )

    def merge(self, dataset):
        assert self.data.keys() == dataset.data.keys()
        for key in dataset.data:
            for x in dataset.get(key):
                self.get(key).append(x)
        return self

    def split_by_size(self, size: int):
        split = []
        for i in range(self.size() // size):
            split.append(self.extract('{}_' + str(i), size))
        if self.size() > 0:
            self.name += '_' + str(len(split))
            split.append(self)
        return split

    def extract(self, name: str, number: int):
        assert number >= self.size()
        data = {}
        for key in self.data:
            data[key] = self.get(key)[:number]
            self.data[key] = self.data[key][number:]
        return Dataset(name=name.format(self.name), data=data)

    def load_all(self, path: str, prefix: str):
        for file in os.listdir(path):
            if file.startswith(prefix):
                print('load : ' + file)
                self.load(os.path.join(path, file))
        return self

    def save(self):
        size = self.size()
        for key in self.data:
            assert len(self.get(key)) == size

        with tf.python_io.TFRecordWriter('{}_n={}.tfrecords'.format(self.name, size)) as writer:
            for i in range(0, size):
                feature_map = {}
                for key in self.data:
                    feature_data = self.get(key)[i]
                    feature_map[key] = _bytes_feature(feature_data.flatten().tostring())
                    feature_map['{}_shape'.format(key)] = _int64_feature(feature_data.shape)

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=feature_map))
                writer.write(example.SerializeToString())

    def load(self, filename: str):
        record_iterator = tf.python_io.tf_record_iterator(path=filename)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            for key in self.data:
                self.get(key).append(_read_feature(example, key))
        return self

    def get(self, key: str):
        return self.data[key]

    def rename(self, feature_name: str, new_name: str):
        self.data[new_name] = self.data[feature_name]
        del self.data[feature_name]
        return self


def _read_feature(example: tf.train.Example, name: str):
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

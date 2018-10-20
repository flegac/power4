from keras.preprocessing.image import ImageDataGenerator

from src.deep.pipeline.dataset import Dataset
import numpy as np


class MyDataset:
    flow_config = {
        'seed': 2210,
        'shuffle': True
        # 'class_mode': 'categorical'
    }

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def generator(self, width, height, batch_size, **kwargs):
        data_generator = ImageDataGenerator(rescale=1. / 255, **kwargs)
        x = np.array(self.dataset.x)
        y = np.array(self.dataset.y)
        x = np.rollaxis(x, 1, 4)
        return data_generator.flow(x, y,
                                   batch_size=batch_size,
                                   **MyDataset.flow_config
                                   )
        # return data_generator.flow_from_directory(self.path,
        #                                           target_size=(height, width),
        #                                           batch_size=batch_size,
        #                                           **MyDataset.flow_config)

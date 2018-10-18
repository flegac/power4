from keras.preprocessing.image import ImageDataGenerator

from src.deep.pipeline.dataset import Dataset


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
        dataset = self.dataset.read()
        x = dataset['X']
        y = dataset['Y']
        return data_generator.flow(x, y,
                                   batch_size=batch_size,
                                   **MyDataset.flow_config
                                   )
        # return data_generator.flow_from_directory(self.path,
        #                                           target_size=(height, width),
        #                                           batch_size=batch_size,
        #                                           **MyDataset.flow_config)

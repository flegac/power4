import os
import random

from matplotlib import pyplot as plt
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator


class MyDataset:
    flow_config = {
        'seed': 2210,
        'shuffle': True,
        'class_mode': 'categorical'
    }

    def __init__(self, path: str, dir_name: str = None) -> None:
        self.path = path if dir_name is None else os.path.join(path, dir_name)
        self.categories = os.listdir(self.path)

    def generator(self, width, height, batch_size, **kwargs):
        data_generator = ImageDataGenerator(rescale=1. / 255, **kwargs)
        return data_generator.flow_from_directory(self.path,
                                                  target_size=(height, width),
                                                  batch_size=batch_size,
                                                  **MyDataset.flow_config)

    def show(self, category_id: str, n=1):
        category_path = self.category_path(category_id)
        image_files = random.sample(os.listdir(category_path), n)
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(category_path, image_file)
            img = misc.imread(image_path)
            plt.figure(i)
            plt.imshow(img)
            plt.title(image_file)
        plt.show()

    def get_images(self, category_id: str, n=1):
        category_path = self.category_path(category_id)
        image_files = random.sample(os.listdir(category_path), n)
        images = []
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            img = misc.imread(image_path)
            images.append(img)
        return images

    def category_path(self, category_id: str):
        return os.path.join(self.path, category_id)

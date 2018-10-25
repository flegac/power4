import json
import os

from keras import models
from keras.layers import *


class MyModel:
    @staticmethod
    def stack_conv_layers(in_layer, n: int, internal_depth: int = 16):
        layer = in_layer
        for i in range(n):
            layer = Conv2D(internal_depth, kernel_size=(3, 3), padding='same')(layer)
            layer = Activation('relu')(layer)
        return layer

    @staticmethod
    def load(name: str, path: str):
        return MyModel(name, models.load_model(path))

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)

    def __init__(self, name: str, model) -> None:
        self.name = name
        self.model = model
        self.img_width = model.input_shape[1]
        self.img_height = model.input_shape[2]
        self.img_bands = model.input_shape[3]

    def keras_model(self):
        return self.model

    def predict(self, x):
        x = np.rollaxis(np.array([x]), 1, 4)
        return self.model.predict(x, batch_size=1)

    def __str__(self):
        return json.dumps(self.model.to_json(),
                          sort_keys=True,
                          indent=4,
                          separators=(',', ': '))

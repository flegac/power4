import json
import os

from keras import models
from keras.utils import plot_model
import numpy as np

class MyModel:
    workspace = 'E:/Flo/workspaces/models'

    @staticmethod
    def load(name):
        return MyModel(name, models.load_model(MyModel._path(name)))

    def save(self, name=None):
        if name is None:
            name = self.name
        save_path = MyModel._path(name)

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

    def plot(self):
        plot_model(self.model, to_file=os.path.join(MyModel.workspace, self.name + '.png'))

    def path(self):
        return os.path.join(MyModel.workspace, self.name)

    @staticmethod
    def _path(name):
        return os.path.join(MyModel.workspace, name, name + '.h5')

    def __str__(self):
        return json.dumps(self.model.to_json(),
                          sort_keys=True,
                          indent=4,
                          separators=(',', ': '))

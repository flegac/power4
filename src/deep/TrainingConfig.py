import json


class TrainingConfig:

    def __init__(self, epochs, batch_size=32,
                 loss='mean_squared_error',
                 optimizer='sgd',
                 metrics=None) -> None:
        if metrics is None:
            metrics = ['acc']

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.verbose = 2

        self.augmentation = {
            # 'rotation_range': 40,
            # 'width_shift_range': 0.2,
            # 'height_shift_range': 0.2,
            # 'shear_range': 0.2,
            # 'zoom_range': 0.2,
            # 'horizontal_flip': True,
            # 'fill_mode': 'nearest'
        }

    def __str__(self):
        return json.dumps(self.__dict__,
                          sort_keys=True,
                          indent=4,
                          separators=(',', ': '))

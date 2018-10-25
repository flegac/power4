import json


class TrainingConfig:

    def __init__(self, epochs, batch_size=32,
                 loss='mean_squared_error',
                 optimizer='adadelta',
                 metrics=None,
                 verbose=2,
                 callbacks=None) -> None:
        """
        :param epochs:
        :param batch_size:
        :param loss:
            mean_squared_error
            mean_absolute_error
            mean_absolute_percentage_error
            mean_squared_logarithmic_error
            squared_hinge
            hinge
            categorical_hinge
            logcosh
            categorical_crossentropy
            sparse_categorical_crossentropy
            binary_crossentropy
            kullback_leibler_divergence
            poisson
            cosine_proximity

        :param optimizer:
            keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
            keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
            keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
            keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)



        :param metrics:
            keras.metrics.binary_accuracy(y_true, y_pred)
            keras.metrics.categorical_accuracy(y_true, y_pred)
            sparse_categorical_accuracy
            top_k_categorical_accuracy
            sparse_top_k_categorical_accuracy

        """
        if metrics is None:
            # metrics = ['logcosh', 'mse', 'acc', 'mae', 'mape', 'cosine']
            metrics = ['logcosh', 'mse', 'mae', 'acc']

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.verbose = verbose
        self.callbacks = callbacks

        self.pre_processing = {
            # rescale = 1. / 255,
        }

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

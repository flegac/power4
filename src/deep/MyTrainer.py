import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

from src.deep import TrainingConfig
from src.deep.MyModel import MyModel
from src.deep.dataset import Dataset


class MyTrainer:
    def __init__(self, training: Dataset, validation: Dataset = None, use_gpu=False) -> None:
        self.training = training
        self.validation = validation
        self.last_history = None
        self.total_epochs = 0
        self.devices = device_lib.list_local_devices()
        self.gpu_devices = filter(lambda x: 'GPU' in x.device_type, self.devices)
        self.cpu_devices = filter(lambda x: 'CPU' in x.device_type, self.devices)
        self.device = next(self.cpu_devices)

        if use_gpu:
            try:
                self.device = next(self.gpu_devices)
            except:
                print('Error : could not use GPU !')

    def fit_model(self, model: MyModel, training_config: TrainingConfig):
        model.model.compile(loss=training_config.loss,
                            optimizer=training_config.optimizer,
                            metrics=training_config.metrics)

        training_flow = self.training.generator(training_config.batch_size,
                                                **training_config.pre_processing,
                                                **training_config.augmentation)

        validation_flow = None
        if self.validation:
            validation_flow = self.validation.generator(training_config.batch_size,
                                                        **training_config.pre_processing)

        print('using device : {}'.format(str(self.device)))
        with tf.device(self.device.name):
            history = model.model.fit_generator(
                training_flow,
                epochs=training_config.epochs,
                shuffle=True,
                verbose=training_config.verbose,
                validation_data=validation_flow,
                callbacks=training_config.callbacks
            )

        model.save('{}_final'.format(model.name))
        self.last_history = history
        self.total_epochs += training_config.epochs
        self.show_history()

    def show_history(self):
        history = self.last_history
        epochs = range(1, len(history.epoch) + 1)

        try:
            acc = history.history['acc']
            val_acc = history.history['val_acc']

            plt.figure()
            plt.title('Training and validation accuracy')
            plt.plot(epochs, acc, 'red', label='Training acc')
            plt.plot(epochs, val_acc, 'blue', label='Validation acc')
            plt.legend()
            plt.show()
        except:
            print('could not plot accuracy history !')

        try:
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure()
            plt.title('Training and validation loss')
            plt.plot(epochs, loss, 'red', label='Training loss')
            plt.plot(epochs, val_loss, 'blue', label='Validation loss')
            plt.legend()
            plt.show()
        except:
            print('could not plot loss function history !!')

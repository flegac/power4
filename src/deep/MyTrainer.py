import json

import keras
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

from monkey.api.TrainingConfig import TrainingConfig
from src.deep.MyDataset import MyDataset
from src.deep.MyModel import MyModel


class MyTrainer:
    def __init__(self, training: MyDataset, validation: MyDataset) -> None:
        self.training = training
        self.validation = validation
        self.last_history = None
        self.total_epochs = 0
        self.devices = device_lib.list_local_devices()
        self.gpu_devices = filter(lambda x: 'GPU' in x.device_type, self.devices)
        self.cpu_devices = filter(lambda x: 'CPU' in x.device_type, self.devices)
        try:
            self.device = next(self.gpu_devices)
        except:
            self.device = next(self.cpu_devices)

    def fit_model(self, model: MyModel, training_config: TrainingConfig):
        model.model.compile(loss=training_config.loss,
                            optimizer=training_config.optimizer,
                            metrics=training_config.metrics)

        training_flow = self.training.generator(model.img_width, model.img_height,
                                                training_config.batch_size,
                                                **training_config.augmentation)
        validation_flow = self.validation.generator(model.img_width, model.img_height,
                                                    training_config.batch_size)

        file_path = MyModel._path(model.name)

        json_log = open('training_logs.json', mode='wt', buffering=1)
        log_callback = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: json_log.write(
                json.dumps({'epoch': epoch, 'logs': logs}) + '\n'),
            on_train_end=lambda logs: json_log.close()
        )

        print('using device : {}'.format(str(self.device)))
        with tf.device(self.device.name):
            history = model.model.fit_generator(
                training_flow,
                epochs=training_config.epochs,
                shuffle=True,
                verbose=training_config.verbose,
                validation_data=validation_flow,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1),
                    # keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
                    log_callback
                ]
            )
        model.save('{}_final'.format(model.name))
        self.last_history = history
        self.total_epochs += training_config.epochs
        self.show_history()

    def show_history(self):
        history = self.last_history
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.title('Training and validation accuracy')
        plt.plot(epochs, acc, 'red', label='Training acc')
        plt.plot(epochs, val_acc, 'blue', label='Validation acc')
        plt.legend()
        plt.show()

        plt.figure()
        plt.title('Training and validation loss')
        plt.plot(epochs, loss, 'red', label='Training loss')
        plt.plot(epochs, val_loss, 'blue', label='Validation loss')
        plt.legend()
        plt.show()

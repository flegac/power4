import json
import os

import keras
import matplotlib.pylab as plt
import tensorflow as tf
import uuid
from tensorflow.python.client import device_lib

from src.deep import TrainingConfig
from src.deep.MyDataset import MyDataset
from src.deep.MyModel import MyModel


class MyTrainer:
    WORKSPACE = 'E:/Flo/workspaces/.my_deep'

    def __init__(self, training: MyDataset, validation: MyDataset = None, use_gpu=False) -> None:
        self.training_id = str(uuid.uuid4())
        self.workspace = os.path.join(MyTrainer.WORKSPACE, self.training_id)
        self.training_logs_path = os.path.join(self.workspace, 'training_logs.json')
        self.model_backup_path = os.path.join(self.workspace, 'models/p4_model_loss={val_loss:3f}.h5')
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_backup_path), exist_ok=True)

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

        training_logs = open(self.training_logs_path, mode='wt', buffering=1)
        print('using device : {}'.format(str(self.device)))
        callbacks = [
            *training_config.callbacks,
            keras.callbacks.ModelCheckpoint(self.model_backup_path,
                                            monitor='val_loss',
                                            verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=5),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: training_logs.write(
                    json.dumps({'epoch': epoch, 'logs': logs}) + '\n'),
                on_train_end=lambda logs: training_logs.close()
            )
        ]
        with tf.device(self.device.name):
            history = model.model.fit_generator(
                training_flow,
                epochs=training_config.epochs,
                shuffle=True,
                verbose=training_config.verbose,
                validation_data=validation_flow,
                callbacks=callbacks
            )
        self.last_history = history
        self.total_epochs += training_config.epochs
        self.save_training(model)

    def _plot_measure(self, model: MyModel, measure: str):
        try:
            history = self.last_history
            epochs = range(1, len(history.epoch) + 1)

            loss = history.history[measure]
            val_loss = history.history['val_{}'.format(measure)]

            plt.figure()
            plt.title('{} : {}'.format(model.name, measure))
            plt.plot(epochs, loss, 'red', label='Training {}'.format(measure))
            plt.plot(epochs, val_loss, 'blue', label='Validation {}'.format(measure))
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(self.workspace, '{}_{}.png'.format(model.name, measure)))
        except:
            print('could not plot {} history !'.format(measure))

    def save_training(self, model: MyModel):
        model.save(os.path.join(self.workspace, '{}_final'.format(model.name)))
        # plot_model(model.keras_model(), to_file=os.path.join(self.workspace, model.name + '.png'))
        self._plot_measure(model, 'mean_squared_error')
        self._plot_measure(model, 'logcosh')

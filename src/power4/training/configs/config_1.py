import keras
import json

from src.deep.TrainingConfig import TrainingConfig

json_log = open('training_logs.json', mode='wt', buffering=1)
training_config = TrainingConfig(
    epochs=500,
    batch_size=1024,
    loss='logcosh',
    optimizer='adadelta',
    # optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
    verbose=2,
    callbacks=[
        keras.callbacks.ModelCheckpoint('../data/models/p4_model_loss={val_loss:3f}.h5',
                                        monitor='val_loss',
                                        verbose=0, save_best_only=True,
                                        save_weights_only=False, mode='auto', period=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto'),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: json_log.write(
                json.dumps({'epoch': epoch, 'logs': logs}) + '\n'),
            on_train_end=lambda logs: json_log.close()
        )
    ]
)

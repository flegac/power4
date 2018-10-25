import keras

from src.deep.TrainingConfig import TrainingConfig

training_config = TrainingConfig(
    epochs=500,
    batch_size=1024,
    loss='mean_squared_error',
    optimizer='adadelta',
    # optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
    verbose=2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    ]
)

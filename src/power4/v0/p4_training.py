from keras import optimizers

from src.deep.MyDataset import MyDataset
from src.deep.MyTrainer import MyTrainer
from src.deep.TrainingConfig import TrainingConfig
from src.deep.pipeline.dataset import Dataset
from src.power4.v0.P4Model import P4Model

training_config = TrainingConfig(epochs=1000,
                                 batch_size=32,
                                 optimizer=optimizers.SGD(lr=1e-6, momentum=0.9)
                                 )
training_config.verbose = 2

filename = 'p4_dataset_0000.tfrecords'

model = P4Model.create()
model.model.summary()

training = MyDataset(Dataset(filename))
trainer = MyTrainer(training)

trainer.fit_model(model, training_config)

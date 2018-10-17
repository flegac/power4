import os
import sys

# training configuration -------------------------------------------------------------------
from src.deep.MyDataset import MyDataset
from src.deep.MyModel import MyModel
from src.deep.MyTrainer import MyTrainer
from src.deep.TrainingConfig import TrainingConfig
from src.power4.v0.P4Model import P4Model


def save_training(config: TrainingConfig):
    try:
        path = 'E:/Flo/workspaces/models/train_config'
        os.makedirs(path)
        with open(os.path.join(path, 'training_config.json'), 'w') as file:
            file.write(str(config))
        print(training_config)
    except Exception as e:
        print('oups : {}'.format(e))


training_config = TrainingConfig(epochs=200,
                                 batch_size=64,
                                 optimizer='adam'
                                 # optimizer=optimizers.SGD(lr=1e-4, momentum=0.9)
                                 )
training_config.verbose = 1

save_training(training_config)


# train the model -------------------------------------------------------------------------

def save_model(model: MyModel):
    model_path = model.path()
    os.makedirs(model_path)
    with open(os.path.join(model_path, 'model_summary.txt'), 'w') as file:
        orig_stdout = sys.stdout
        sys.stdout = file
        model.model.summary()
        sys.stdout = orig_stdout


model = P4Model.create()

save_model(model.path())
model.model.summary()

ROOT_PATH = 'E:/Flo/workspaces/p4_models'
training = MyDataset(ROOT_PATH, 'training')
validation = MyDataset(ROOT_PATH, 'validation')
trainer = MyTrainer(training, validation)

trainer.fit_model(model, training_config)

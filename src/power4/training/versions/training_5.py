from src.deep.MyTrainer import MyTrainer
from src.power4.training.configs.config_0 import training_config
from src.power4.training.configs.dataset_0 import training, validation
from src.power4.training.models.model_5 import P4Model

model = P4Model.create()
# model = P4Model.load('p4_model')
model.model.summary()

trainer = MyTrainer(training, validation, use_gpu=True)
trainer.fit_model(model, training_config)

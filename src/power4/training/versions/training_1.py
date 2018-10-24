from src.deep.MyTrainer import MyTrainer
from src.power4.training.configs.config_0 import training_config
from src.power4.training.configs.dataset_0 import training, validation
from src.power4.training.models.basic_model import P4Model

model = P4Model.create('v1', convlayer_number=8, conv_layer_depth=32, dense_size=64)
model.model.summary()

trainer = MyTrainer(training, validation, use_gpu=True)
trainer.fit_model(model, training_config)

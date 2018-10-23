from src.deep.MyTrainer import MyTrainer
from src.power4.training.configs.config_0 import training_config
from src.power4.training.configs.dataset_0 import training, validation
from src.power4.training.models.model_0 import P4Model

model = P4Model.create('v2', convlayer_number=6, conv_layer_depth=64, dense_size=128)
model.model.summary()

trainer = MyTrainer(training, validation, use_gpu=True)
trainer.fit_model(model, training_config)

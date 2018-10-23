from keras.layers import *
from keras.models import Model

from src.deep.MyModel import MyModel
from src.power4.P4Board import P4Board


class P4Model(MyModel):

    @staticmethod
    def create(name: str, convlayer_number: int, conv_layer_depth: int, dense_size: int):
        img_width = P4Board.GRID_WIDTH
        img_height = P4Board.GRID_HEIGHT
        img_bands = 4
        output_size = 1

        input_layer = Input(shape=(img_width, img_height, img_bands,))

        tmp = MyModel.stack_conv_layers(input_layer, n=convlayer_number, internal_depth=conv_layer_depth)

        tmp = Flatten()(tmp)
        tmp = Dense(dense_size)(tmp)
        tmp = Activation('relu')(tmp)

        tmp = Dense(output_size)(tmp)
        output_layer = Activation('linear')(tmp)

        model = Model(input_layer, output_layer)

        return P4Model('p4_model_' + name, model)

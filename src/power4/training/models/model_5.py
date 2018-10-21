from keras.layers import *
from keras.models import Model

from src.deep.MyModel import MyModel
from src.power4.P4Board import P4Board


class P4Model(MyModel):
    @staticmethod
    def create():
        def stack_layers(input_layer, n: int, internal_depth: int = 16):
            layer = input_layer
            for i in range(n):
                layer = Conv2D(internal_depth, kernel_size=(3, 3), padding='same')(layer)
                layer = Activation('relu')(layer)
            return layer

        img_width = P4Board.GRID_WIDTH
        img_height = P4Board.GRID_HEIGHT
        img_bands = 4

        output_size = 1

        input_layer = Input(shape=(img_width, img_height, img_bands,))

        tmp = stack_layers(input_layer, n=5, internal_depth=16)
        tmp = stack_layers(tmp, n=5, internal_depth=32)
        tmp = stack_layers(tmp, n=5, internal_depth=64)
        tmp = stack_layers(tmp, n=5, internal_depth=32)
        tmp = stack_layers(tmp, n=5, internal_depth=16)

        tmp = Flatten()(tmp)
        tmp = Dense(16)(tmp)
        tmp = Activation('relu')(tmp)

        tmp = Dense(output_size)(tmp)
        output_layer = Activation('linear')(tmp)

        model = Model(input_layer, output_layer)

        return P4Model('p4_model_v5', model)

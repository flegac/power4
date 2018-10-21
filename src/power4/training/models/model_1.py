from keras.layers import *
from keras.models import Model

from src.deep.MyModel import MyModel
from src.power4.P4Board import P4Board


class P4Model(MyModel):
    @staticmethod
    def create():
        img_width = P4Board.GRID_WIDTH
        img_height = P4Board.GRID_HEIGHT
        img_bands = 4

        output_size = 1

        input_layer = Input(shape=(img_width, img_height, img_bands,))

        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)

        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(64, kernel_size=(3, 3), padding='same')(tmp)
        tmp = Activation('relu')(tmp)

        tmp = Flatten()(tmp)
        tmp = Dense(64)(tmp)
        tmp = Activation('relu')(tmp)

        tmp = Dense(output_size)(tmp)
        output_layer = Activation('linear')(tmp)

        model = Model(input_layer, output_layer)

        return P4Model('p4_model_v1', model)

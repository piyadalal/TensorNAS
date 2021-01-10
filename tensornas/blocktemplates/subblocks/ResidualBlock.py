from enum import Enum, auto
from tensorflow import keras

from tensornas.core.block import Block
from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)


class SubBlockTypes(Enum):
    TWOD_CLASSIFICATION = auto()


class ResidualBlock(Block):
    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    def __init__(self, input_shape, parent_block, class_count, layer_type):
        self.class_count = class_count

        super().__init__(input_shape, parent_block, layer_type)

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.TWOD_CLASSIFICATION:
            return [
                TwoDClassificationBlock(
                    input_shape=input_shape, parent_block=self, class_count=self.class_count, layer_type=layer_type
                )
            ]
        return []

    def get_keras_layers(self):
        inp = keras.Input(shape=self.input_shape)
        tmp = inp
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers()(tmp)
        return keras.layers.Add([inp, tmp])

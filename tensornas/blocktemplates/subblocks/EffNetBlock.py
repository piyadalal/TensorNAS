from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers

class SubBlockTypes(Enum):
    CONV2D = auto()
    MAXPOOL = auto()
    DEPTHWISE_CONV2D_1_n = auto()
    DEPTHWISE_CONV2D_n_1 = auto()
    POINTWISE_CONV2D=auto()


class EffNetBlock(Block):

    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.POINTWISE_CONV2D,
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):

            return[
            [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.DEPTHWISECONV1N,
                )
            ]
            [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.MAXPOOL,
                )
            ]
            [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.DEPTHWISECONVN1,
                )
            ]
        ]
    def generate_constrained_output_sub_blocks(self, input_shape):
        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.CONV2D,
            )
        ]

    def get_keras_layers(self):
        array = None
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            if not array:
                array = sb.get_keras_layers()
            else:
                array = sb.get_keras_layers()(array)
        return array

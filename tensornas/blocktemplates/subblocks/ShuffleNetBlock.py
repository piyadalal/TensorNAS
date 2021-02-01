from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers


class SubBlockTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    DEPTHWISECONV2D = auto()
    #POINTWISECONV2D =auto()
    GROUPEDCONV2D = auto()


class ShuffleNetBlock(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.GROUPEDCONV2D,
            )
            #to be included is channel shuflle operation
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.DEPTHWISECONV2D:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.DEPTHWISECONV2D,
                )
            ]
        elif layer_type == self.SUB_BLOCK_TYPES.GROUPEDCONV2D:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.GROUPEDCONV2D,
                )
                #channel -shuffle
            ]
        return []

    def generate_constrained_output_sub_blocks(self, input_shape):
        return[
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.GROUPEDCONV2D,
            )
        ]


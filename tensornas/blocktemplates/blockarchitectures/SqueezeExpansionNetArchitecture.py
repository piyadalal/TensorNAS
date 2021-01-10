from enum import Enum, auto
from tensornas.layers import SupportedLayers
from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)
from tensornas.core.layerblock import LayerBlock
from tensornas.blocktemplates.subblocks.SqueezeExpansionBlock import SqueezeExpansionBlock
from tensornas.core.blockarchitecture import BlockArchitecture

class SqueezeExpansionNetArchitectureSubBlocks(Enum):
    SQUEEZE_EXPANSION_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()

class SqueezeExpansionNetBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = SqueezeExpansionNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None, layer_type=None)

    def validate(self, repair):
        ret = True
        """should add a squeezeBlock for auto selection to tensornas.blocktemplates.subblocks """
        if not isinstance(self.output_blocks[0], TwoDClassificationBlock):
            ret = False

        return ret

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.CONV2D
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        return [
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
                layer_type=self.SUB_BLOCK_TYPES.CLASSIFICATION_BLOCK,
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        return [
            SqueezeExpansionBlock(
                input_shape=input_shape, parent_block=self, layer_type=layer_type
            )
        ]
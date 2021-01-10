from enum import Enum, auto

from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)
from tensornas.blocktemplates.subblocks.ZFNetBlock import (
    ZFNetBlock,
)
from tensornas.core.blockarchitecture import BlockArchitecture


class ZFNetArchitectureSubBlocks(Enum):
    ZFNET_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()
    #filter size 7*7
    #stride 2


class ClassificationBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = ZFNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None, layer_type=None)

    def validate(self, repair):
        ret = True
        if not isinstance(self.output_blocks[-1], TwoDClassificationBlock):
            ret = False
        return ret

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
        if layer_type == self.SUB_BLOCK_TYPES.ZFNET_BLOCK:
            return [
                ZFNetBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        return []

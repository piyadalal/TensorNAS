from tensornas.blockarchitecture import BlockArchitecture
from tensornas.blocks import ClassificationBlock
from tensornas.blocks import FeatureExtractionBlock
from enum import Enum, auto


class TopLevelBlockTypes(Enum):
    CLASSIFICATION_BLOCK = auto()
    FEATURE_EXTRACTION_BLOCK = auto()


class ClassificationBlockArchitecture(BlockArchitecture):

    MAX_SUB_BLOCKS = 5
    SUB_BLOCK_TYPES = TopLevelBlockTypes

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape)

    def validate(self):
        ret = True
        if not isinstance(self.sub_blocks[-1], ClassificationBlock):
            ret = False
        return ret

    def generate_constrained_input_sub_blocks(self):
        self.sub_blocks.append(
            ClassificationBlock(
                input_shape=self.input_shape,
                parent_block=self,
                class_count=self.class_count,
            )
        )

    def generate_random_sub_block(self, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.CLASSIFICATION_BLOCK.value:
            return ClassificationBlock(input_shape=self.input_shape, parent_block=self)
        elif layer_type == self.SUB_BLOCK_TYPES.FEATURE_EXTRACTION_BLOCK.value:
            return FeatureExtractionBlock(
                input_shape=self.input_shape, parent_block=self
            )
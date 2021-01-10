from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers

class SqueezeExpansionBlockLayerTypes(Enum):

    """Contains Global Average Pool layer and then Fully connected layers with ReLu types and output dense layer with sigmoid activation function"""

    GLOBAL_AVERAGE_POOLING2D = auto()
    HIDDENDENSE = auto()
    OUTPUTDENSE = auto()

class SqueezeExpansionBlock(Block):

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SqueezeExpansionBlockLayerTypes



    def generate_random_sub_block(self, input_shape, layer_type):

        pooling= LayerBlock(
                input_shape=None, parent_block=self, layer_type=SupportedLayers.GLOBALAVERAGEPOOL2D
            )
        Dense =  LayerBlock(
                input_shape=None,
                parent_block=self,
                layer_type=SupportedLayers.HIDDENDENSE,
            )
        return [pooling,Dense]

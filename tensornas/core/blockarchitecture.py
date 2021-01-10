import numpy as np
import tensorflow as tf
import keras
import pydot
from ann_visualizer.visualize import ann_viz
import pydotplus
from pydotplus import graphviz
from keras_sequential_ascii import keras2ascii

from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot
from tensornas.core.block import Block
from tensorflow.python.keras.layers import SeparableConv2D,Conv2D,DepthwiseConv2D,GlobalAveragePooling2D, MaxPool3D,Dense,Flatten, Dropout, MaxPooling2D,DepthwiseConv2D


import visualkeras
from collections import defaultdict

class BlockArchitecture(Block):
    """
    A block architectures, eg. a classification architecture is one that provides a specified
    number of probability outputs that are used in the classification of some input.
    The abstract block architecture class defines the methods that must be implemented to allow for a type of block
    architecture to be created, namely what sort of sub-blocks the block architecture can generate.
    """


    color_map = defaultdict(dict)
    color_map[Conv2D]['fill'] = 'orange'
    color_map[SeparableConv2D]['fill'] = 'gray'
    color_map[DepthwiseConv2D]['fill'] = 'yellow'
    color_map[GlobalAveragePooling2D]['fill'] = 'pink'
    color_map[MaxPooling2D]['fill'] = 'red'
    color_map[Dense]['fill'] = 'green'
    color_map[Flatten]['fill'] = 'teal'
    color_map[Dropout]['fill'] ='black'

    def get_keras_model(self, optimizer, loss, metrics):
        layers = self.get_keras_layers()
        model = tf.keras.Sequential()

        for layer in layers:
            model.add(layer)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        #model.add(visualkeras.SpacingDummyLayer(spacing=100))
        #visualkeras.layered_view(model, spacing=0, to_file="model.png")
        keras2ascii(model)
        #plot_model(model, to_file="Output.png")
        #ann_viz(model, title="Artificial Neural network - Model Visualization")
        return model

    def evaluate(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs,
        steps,
        batch_size,
        optimizer,
        loss,
        metrics,
    ):
        model = self.get_keras_model(optimizer=optimizer, loss=loss, metrics=metrics)
        # model.summary()
        try:
            model.fit(
                x=train_data,
                y=train_labels,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps,
                verbose=1,
            )
        except Exception as e:
            import math

            print("Error fitting model, {}".format(e))
            return [math.inf, math.inf, 0]
        ret = [
            int(
                np.sum(
                    [tf.keras.backend.count_params(p) for p in model.trainable_weights]
                )
            )
            + int(
                np.sum(
                    [
                        tf.keras.backend.count_params(p)
                        for p in model.non_trainable_weights
                    ]
                )
            ),
            model.evaluate(test_data, test_labels)[1] * 100,
        ]
        return ret

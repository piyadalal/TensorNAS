import tensorflow as tf
from tensornas.layers.MaxPool import Layer

class Layer(Layer):
    def get_keras_layer(self):
        return [
            tf.keras.layers.GlobalAveragePooling2D(data_format=None)
            ]

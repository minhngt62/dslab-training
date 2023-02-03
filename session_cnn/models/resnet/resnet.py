import tensorflow as tf
from tensorflow import keras
from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

class Resnet34(keras.Model):
    def __init__(
            self,
            output_dim,
            input_shape=[224, 224, 3],
            **kwargs     
        ):
        super().__init__(**kwargs)
        self.low_conv = [
            keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False,
                                input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        ]
        
        prev_filters = 64
        self.rus = []
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.rus.append(ResidualUnit(filters, strides))
            prev_filters = filters
        
        self.avg_pool = keras.layers.GlobalAvgPool2D()
        self.out = [
            keras.layers.Flatten(),
            keras.layers.Dense(output_dim, activation="softmax")
        ]
    
    def call(self, inputs):
        for unit in self.low_conv:
            z = unit(inputs)
        for ru in self.rus:
            z = ru(z)
        
        z = self.avg_pool(z)
        for unit in self.out:
            z = unit(z)
        return z 



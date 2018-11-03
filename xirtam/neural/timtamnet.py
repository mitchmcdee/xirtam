"""
Module containing TimTamNet.
"""
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Input, Model


def fire_module(x, num_squeeze_filters, num_expand_filters):
    """
    Fire module from SqueezeNet.
    """
    x = Conv2D(num_squeeze_filters, (1, 1), activation="relu")(x)
    left = Conv2D(num_expand_filters, (1, 1), activation="relu")(x)
    right = Conv2D(num_expand_filters, (3, 3), padding='same', activation="relu")(x)
    return concatenate([left, right])


def TimTamNet(input_shape=(128, 128, 1)):
    """
    TimTamNet is a Fully Convolutional Network (FCN) designed to learn segmented
    regions of an image.
    """
    inputs = Input(shape=input_shape)
    x = fire_module(inputs, 2, 4)
    x = MaxPooling2D((2, 2))(x)
    x = fire_module(x, 2, 4)
    x = MaxPooling2D((2, 2))(x)
    x = fire_module(x, 4, 4)
    x = MaxPooling2D((2, 2))(x)
    x = fire_module(x, 2, 4)
    x = UpSampling2D((2, 2))(x)
    x = fire_module(x, 4, 4)
    x = UpSampling2D((2, 2))(x)
    x = fire_module(x, 2, 4)
    x = UpSampling2D((2, 2))(x)
    x = fire_module(x, 2, 4)
    x = Conv2D(1, (1, 1), activation="sigmoid")(x)
    return Model(inputs, x, name='timtamnet')

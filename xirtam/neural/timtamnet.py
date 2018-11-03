"""
Module containing TimTamNet.
"""
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Input, Model


def fire_module(fire_id, num_squeeze, num_expand):
    """
    Fire module from SqueezeNet.
    """
    name = f"fire_{fire_id}"
    squeeze = Conv2D(num_squeeze, (1, 1), activation="relu", name=f"{name}/squeeze_1x1")
    left = Conv2D(num_expand, (3, 3), activation="relu", name=f"{name}/expand_3x3", padding="same")
    right = Conv2D(num_expand, (1, 1), activation="relu", name=f"{name}/expand_1x1")

    def fire_module_wrapper(x):
        squeezed = squeeze(x)
        return concatenate([left(squeezed), right(squeezed)])

    return fire_module_wrapper


def TimTamNet(input_shape=(128, 128, 1)):
    """
    TimTamNet is a Fully Convolutional Network (FCN) designed to learn segmented
    regions of an image.
    """
    inputs = Input(shape=input_shape)
    x = fire_module(fire_id=1, num_squeeze=2, num_expand=4)(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = fire_module(fire_id=2, num_squeeze=2, num_expand=4)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = fire_module(fire_id=3, num_squeeze=4, num_expand=4)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = fire_module(fire_id=4, num_squeeze=2, num_expand=4)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = fire_module(fire_id=5, num_squeeze=4, num_expand=4)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = fire_module(fire_id=6, num_squeeze=2, num_expand=4)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = fire_module(fire_id=7, num_squeeze=2, num_expand=4)(x)
    x = Conv2D(1, (1, 1), activation="sigmoid")(x)
    return Model(inputs, x, name="timtamnet")

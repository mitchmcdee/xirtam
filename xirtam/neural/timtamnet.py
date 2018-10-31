"""
Module containing TimTamNet.
"""
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential


def TimTamNet(input_shape=(128, 128, 1)):
    """
    TimTamNet is a Fully Convolutional Network (FCN) designed to learn segmented
    regions of an image.
    """
    timtamnet = Sequential(name="timtamnet")
    timtamnet.add(Conv2D(4, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    timtamnet.add(Conv2D(4, (3, 3), padding="same", activation="relu", dilation_rate=2))
    timtamnet.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    timtamnet.add(Conv2D(8, (3, 3), padding="same", activation="relu", dilation_rate=2))
    timtamnet.add(Conv2DTranspose(8, (3, 3), padding="same", activation="relu"))
    timtamnet.add(Conv2DTranspose(8, (3, 3), padding="same", activation="relu", dilation_rate=2))
    timtamnet.add(Conv2DTranspose(4, (3, 3), padding="same", activation="relu"))
    timtamnet.add(Conv2DTranspose(4, (3, 3), padding="same", activation="relu", dilation_rate=2))
    timtamnet.add(Conv2D(1, (1, 1), padding="same", activation="sigmoid"))
    return timtamnet

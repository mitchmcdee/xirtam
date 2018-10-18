"""
Module containing TimTamNet.
"""
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential


class TimTamNet(Sequential):
    """
    TimTamNet is a Fully Convolutional Network (FCN) designed to learn segmented
    regions of an image.
    """

    def __init__(self, input_shape=(128, 128, 1)) -> None:
        super().__init__(name="timtamnet")
        self.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        self.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        self.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        self.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        self.add(UpSampling2D((2, 2)))
        self.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        self.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        self.add(UpSampling2D((2, 2)))
        self.add(Conv2D(1, (1, 1), padding="same", activation="sigmoid"))

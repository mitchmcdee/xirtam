import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import skimage.transform
import skimage.color
import skimage.io
from tqdm import tqdm


def seg_data_generator(stride, n_classes, img_dir, img_list, preprocess=True):
    LUT = np.eye(n_classes)
    x_ = []
    y_ = []
    for img_id in tqdm(img_list):
        # load image
        img_path = img_dir + img_id
        x = skimage.io.imread(img_path)

        # load label
        label_path = img_dir + "regions.bmp"
        y = skimage.io.imread(label_path)  # interprets the image as a colour image

        # only yield is the images exist
        is_img = type(x) is np.ndarray and type(y) is np.ndarray
        is_empty = len(x.shape) == 0 or len(y.shape) == 0
        if not is_img or is_empty:
            continue

        # deal with gray value images
        if len(x.shape) == 2:
            x = skimage.color.gray2rgb(x)

        # only take one channel
        if len(y.shape) > 2:
            y = y[..., 0]

        # prepare for NN
        x = np.array(x, dtype="float")
        if preprocess == True:
            x = preprocess_input(x)

        y = LUT[y]
        x_.append(x)
        y_.append(y)
    return np.stack(x_, axis=0), np.stack(y_, axis=0)

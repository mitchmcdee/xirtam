import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from model_fcns import resnet50_fcn, resnet50_16s_fcn, resnet50_8s_fcn
from data_generator import seg_data_generator
import utils

utils.config_tf()

model_input_dir = "./work/fcn_models/testing/"
img_dir = "./data/out/world-5546682508642403194/robot--4209387126734636757/"
net = "resnet50"
n_classes = 3

if net == "resnet50":
    model, stride = resnet50_fcn(n_classes)

if net == "resnet50_16s":
    model, stride = resnet50_16s_fcn(n_classes)

if net == "resnet50_8s":
    model, stride = resnet50_8s_fcn(n_classes)

if model_input_dir != "":
    model.load_weights(model_input_dir + "best_weights.hdf5")

img_id = 0
n_rows = 5
n_cols = 2

# Load images
img_list = os.listdir(img_dir)
random.shuffle(img_list)
img_list = img_list[:n_rows * n_cols]
x_test, y_test = seg_data_generator(stride, n_classes, img_dir, img_list, preprocess=False)
x_test = iter(x_test)
y_test = iter(y_test)

# Visualise in tiled plot
fig = plt.figure(figsize=(15, 15))

for idx in range(n_rows):
    for row_img in range(n_cols):
        x = next(x_test)
        y = next(y_test)
        x = x[np.newaxis, ...]  # make it a 4D tensor
        y = y[np.newaxis, ...]  # make it a 4D tensor
        y = np.argmax(y, axis=-1)[0]

        pred = model.predict(x, batch_size=1)[0]
        pred = np.argmax(pred, axis=-1)

        # Visualise result
        fig.add_subplot(n_rows, 6, idx * 6 + row_img * 3 + 1)
        plt.imshow(x[0])
        plt.gray()

        fig.add_subplot(n_rows, 6, idx * 6 + row_img * 3 + 2)
        plt.imshow(y, vmin=0, vmax=n_classes - 1)
        plt.gray()

        fig.add_subplot(n_rows, 6, idx * 6 + row_img * 3 + 3)
        plt.imshow(pred, vmin=0, vmax=n_classes - 1)
        plt.gray()

plt.show()

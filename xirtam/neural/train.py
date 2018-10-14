import numpy as np
import random
import os
import argparse

import pandas as pd

import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

from model_fcns import resnet50_fcn, testnet_fcn, resnet50_16s_fcn, resnet50_8s_fcn
from data_generator import seg_data_generator
from loss_fcns import fcn_xent, fcn_xent_nobg, pixel_acc, mean_acc
import utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--train_test_split", help="Ratio of train:test", type=float, default=0.8
    )

    parser.add_argument(
        "-n", "--net", help="Net to train (testnet, resnet50 or resnet50_16s)", default="resnet50"
    )

    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=1)

    parser.add_argument("-b", "--batch_size", help="Size of batch", type=int, default=256)

    parser.add_argument("-o", "--opt", help="Optimizer", default="SGD")

    parser.add_argument("-d", "--img_dir", help="Directory containing the images", required=True)

    parser.add_argument("-lr", "--learning_rate", help="Initial learning rate", default=0.01)

    parser.add_argument("-mi", "--model_input", help="Init with model")

    parser.add_argument(
        "-mo",
        "--model_output",
        help="Where to save the trained model?",
        default="./work/fcn_models/",
    )

    parser.add_argument("-id", "--exp_id", help="Experiment id", required=True)

    return parser.parse_args()


######################################################################################

args = parse_args()
utils.config_tf()

# create experimental directory
model_output_dir = args.model_output + args.exp_id
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

# background, valid, invalid
n_classes = 3

# create model
if args.net == "resnet50":
    model, stride = resnet50_fcn(n_classes)

if args.net == "resnet50_16s":
    model, stride = resnet50_16s_fcn(n_classes, args.model_input)

if args.net == "resnet50_8s":
    model, stride = resnet50_8s_fcn(n_classes, args.model_input)

if args.net == "testnet":
    model, stride = testnet_fcn(n_classes)

# create data generators
img_list = os.listdir(args.img_dir)
random.shuffle(img_list)
num_train = int(args.train_test_split * len(img_list))
num_test = int((1 - args.train_test_split) * len(img_list))
img_list_train = img_list[:num_train]
img_list_test = img_list[num_train:]

x_train, y_train = seg_data_generator(stride, n_classes, args.img_dir, img_list_train)
x_test, y_test = seg_data_generator(stride, n_classes, args.img_dir, img_list_test)

# callbacks
filepath = model_output_dir + "/best_weights.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

tb = TensorBoard(log_dir=model_output_dir, histogram_freq=2, write_graph=False)

plateau = ReduceLROnPlateau(patience=5)

callbacks_list = [checkpoint, tb, plateau]


learning_rate = float(args.learning_rate)
if args.opt == "Adam":
    opt = Adam(lr=learning_rate)
elif args.opt == "SGD":
    opt = SGD(lr=learning_rate, momentum=0.9)
elif args.opt == "SGD_Aggr":
    opt = SGD(lr=learning_rate, momentum=0.99)

# model.compile(optimizer=opt, loss=fcn_xent_nobg, metrics=[mean_acc])
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[mean_acc])

print(model.summary())

model.fit(
    x_train,
    y_train,
    callbacks=callbacks_list,
    validation_data=(x_test, y_test),
    verbose=1,
    epochs=args.epochs,
    batch_size=args.batch_size,
    shuffle=True,
)
model.save(model_input_dir + "final_weights.hdf5")

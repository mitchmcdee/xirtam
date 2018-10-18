"""
Module containing functions pertaining to the training of TimTamNet.
"""
import os
import argparse
from keras.callbacks import ModelCheckpoint, TensorBoard
from timtamnet import TimTamNet
from utils import get_data


def parse_args():
    """
    Parses and extracts training CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--robot_dir_path",
        type=str,
        help="Path to base robot directory containing training/test data",
        required=True
    )
    parser.add_argument(
        "-l",
        "--log_dir_path",
        type=str,
        help="Path to base logs directory containing tensorboard logs",
        default="./out/logs",
    )
    parser.add_argument(
        "-m",
        "--model_dir_path",
        type=str,
        help="Path to base models directory for outputting model weights",
        default="./out/models",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        help="Optimizer algorithm to use during model training",
        default="adam",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Maximum number of epochs to train model for before stopping",
        default=50,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size to use for images during model training",
        default=128,
    )
    return parser.parse_args()


def train(robot_dir_path, log_dir_path, model_dir_path, epochs, batch_size, optimizer):
    """
    Trains TimTamNet and exports model state throughout the training process.
    """
    # Make model directory if it doesn't already exist.
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model_output_path = os.path.join(model_dir_path, "weights.{epoch:02d}-{val_loss:.6f}.hdf5")
    # Setup callbacks.
    tensorboard = TensorBoard(log_dir=log_dir_path, write_graph=False)
    checkpoint = ModelCheckpoint(
        model_output_path, verbose=1, mode="min", save_best_only=True, save_weights_only=True
    )
    callbacks_list = [checkpoint, tensorboard]
    # Get training/test data.
    x_train, x_test, y_train, y_test = get_data(robot_dir_path)
    input_shape = x_train.shape[1:]
    # Create, compile and fit TimTamNet.
    model = TimTamNet(input_shape=input_shape)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    model.summary()
    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
    )


if __name__ == "__main__":
    training_args = parse_args()
    train(**vars(training_args))

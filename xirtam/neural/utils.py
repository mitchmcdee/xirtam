"""
Module containing helpful utility functions pertaining to the training and debugging of TimTamNet.
"""
from numpy import array, reshape
from os import listdir
from os.path import splitext, join
from tqdm import tqdm
from skimage.io import imread


def get_data(robot_directory):
    """
    Retrives the training/test data from the given robot directory.
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for data_type in ("train", "test"):
        data_folder = join(robot_directory, data_type)
        data_x = x_train if data_type == "train" else x_test
        data_y = y_train if data_type == "train" else y_test
        # Sort to ensure deterministic results.
        for world_name in tqdm(list(sorted(listdir(data_folder)))):
            world_dir = join(data_folder, world_name)
            world_x = []
            world_y = []
            # Sort to ensure deterministic results.
            for file_name in list(sorted(listdir(world_dir))):
                _, extension = splitext(file_name)
                if extension != ".bmp":
                    continue
                image = imread(join(world_dir, file_name), as_gray=True)
                if file_name == "regions.bmp":
                    world_y.append(image)
                else:
                    world_x.append(image)
            world_y *= len(world_x)
            data_x.extend(world_x)
            data_y.extend(world_y)
    # Normalise the colour value.
    x_train = array(x_train).astype("float32") / 255
    y_train = array(y_train).astype("float32") / 255
    x_test = array(x_test).astype("float32") / 255
    y_test = array(y_test).astype("float32") / 255
    # Reshape to contain the number of channels (gray = 1 channel).
    x_train = reshape(x_train, (*x_train.shape, 1))
    y_train = reshape(y_train, (*y_train.shape, 1))
    x_test = reshape(x_test, (*x_test.shape, 1))
    y_test = reshape(y_test, (*y_test.shape, 1))
    return x_train, x_test, y_train, y_test

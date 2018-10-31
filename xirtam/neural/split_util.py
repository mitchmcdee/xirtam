"""
Module containing helpful utility functions to split training/test data.
"""
import os
from argparse import ArgumentParser
from random import random
from tqdm import tqdm


def parse_args():
    """
    Parses data splitter CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("-r", "--robot_dir", type=str, help="Path to robot directory", required=True)
    parser.add_argument("-t", "--test_split", type=float, help="Test set split", default=0.1)
    return parser.parse_args()


def split_data(robot_dir, test_split):
    """
    Splits the robot data into training and test sets.
    """
    # Make train and test folders if they don't already exist.
    for data_type in ("test", "train"):
        data_dir = os.path.join(robot_dir, data_type)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    # Move worlds across.
    for world_name in tqdm(list(sorted(os.listdir(robot_dir)))):
        if "world" not in world_name:
            continue
        data_type = "test" if random() <= test_split else "train"
        world_dir = os.path.join(robot_dir, world_name)
        data_world_dir = os.path.join(robot_dir, data_type, world_name)
        os.rename(world_dir, data_world_dir)


if __name__ == "__main__":
    parsed_args = parse_args()
    split_data(parsed_args.robot_dir, parsed_args.test_split)

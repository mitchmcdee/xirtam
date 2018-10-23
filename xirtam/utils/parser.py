"""
Module containing high level class for parsing simulation CLI parameters.
"""
import argparse


class SimulationParser(argparse.ArgumentParser):
    """
    A custom parser for configuring a simulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        simulation_type_group = self.add_mutually_exclusive_group(required=True)
        simulation_type_group.add_argument(
            "-s", "--simulator", action="store_true", help="Run simulator"
        )
        simulation_type_group.add_argument(
            "-t", "--trainer", action="store_true", help="Run trainer"
        )
        self.add_argument(
            "-g", "--generate_case", action="store_true", help="Generate a random test case"
        )
        self.add_argument(
            "-r",
            "--robot_path",
            type=str,
            help="Path to robot file",
            default="./examples/magneto.robot",
        )
        self.add_argument(
            "-o", "--output_path", type=str, help="Path to output folder", default="./out/"
        )
        self.add_argument(
            "-w",
            "--world_path",
            type=str,
            help="Path to world file",
            default="./examples/simple.world",
        )
        self.add_argument(
            "-m",
            "--motion_path",
            type=str,
            help="Path to motion file",
            default="./examples/simple.motion",
        )
        self.add_argument("-n", "--model_path", type=str, help="Path to neural net model file")

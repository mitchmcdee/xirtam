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
            "-s", "--simulator", dest="simulator", action="store_true", help="Run simulator"
        )
        simulation_type_group.add_argument(
            "-t", "--trainer", dest="trainer", action="store_true", help="Run trainer"
        )
        self.add_argument(
            "-w",
            "--world",
            dest="world_path",
            type=str,
            help="Path to world file",
            default="/data/config/test.world",
        )
        self.add_argument(
            "-ro",
            "--robot",
            dest="robot_path",
            type=str,
            help="Path to robot file",
            default="/data/config/test.robot",
        )
        self.add_argument(
            "-m",
            "--motion",
            dest="motion_path",
            type=str,
            help="Path to motion file",
            default="/data/config/test.motion",
        )
        self.add_argument(
            "-o",
            "--output",
            dest="output_path",
            type=str,
            help="Path to generated output folder",
            default="/data/out/",
        )

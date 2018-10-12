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
        simulation_type_group.add_argument('-s', '--simulator', dest='simulator', action='store_true')
        simulation_type_group.add_argument('-t', '--trainer', dest='trainer', action='store_true')
        self.add_argument(
            "-w", "--world", dest="world_path", type=str, required=True, help="Path to world file"
        )
        self.add_argument(
            "-ro", "--robot", dest="robot_path", type=str, required=True, help="Path to robot file"
        )
        self.add_argument(
            "-m",
            "--motion",
            dest="motion_path",
            type=str,
            required=True,
            help="Path to motion file",
        )
        self.add_argument(
            "-p",
            "--policy",
            dest="policy_path",
            type=str,
            required=True,
            help="Path to policy file",
        )
        self.add_argument(
            "-o",
            "--output",
            dest="output_path",
            type=str,
            required=True,
            help="Path to generated output folder",
        )

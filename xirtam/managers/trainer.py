"""
Module containing high level class for controlling policy training.
"""
import sys
import logging
from time import sleep
from model import Model
from settings import LOG_LEVEL
from simulation_parser import SimulationParser
from multiprocessing import Process, cpu_count


# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Trainer:
    """
    A controlling class for training robot policy.
    """

    def __init__(self, trainer_num, world_filepath, robot_filepath, motion_filepath):
        self.model = Model(world_filepath, robot_filepath, motion_filepath)
        LOGGER.info(f"Running trainer #{trainer_num}...")
        self.run()

    def run(self):
        """
        Run the trainer.
        """
        self.model.handle_start()
        while True:
            self.model.update(is_training=True)


class TrainerHandler:
    """
    A controlling class for trainers.
    """

    def __init__(self):
        parsed_args = SimulationParser().parse_args(sys.argv[1:])
        trainer_args = (parsed_args.world_path, parsed_args.robot_path, parsed_args.motion_path)
        num_trainers = max(1, cpu_count() - 1)
        for trainer_num in range(num_trainers):
            trainer = Process(target=Trainer, args=(trainer_num,) + trainer_args)
            trainer.daemon = True
            trainer.start()

    def run(self):
        """
        Run the trainer handler. Essentially sleep and wait for an exit command.
        """
        while True:
            sleep(0.1)


if __name__ == "__main__":
    TrainerHandler().run()

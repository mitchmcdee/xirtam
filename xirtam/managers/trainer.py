"""
Module containing high level class for controlling policy training.
"""
import sys
import signal
import logging
from time import sleep
from multiprocessing import Process, cpu_count
from xirtam.core.model import Model
from xirtam.core.settings import LOG_LEVEL
from xirtam.utils.grid_world_generator import process_generation_args


# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Trainer:
    """
    A controlling class for training robot policy.
    """

    def __init__(self, trainer_num, *args, **kwargs):
        args, kwargs = process_generation_args(trainer_num, *args, **kwargs)
        signal.signal(signal.SIGINT, self.on_close)
        self.model = Model(*args, **kwargs)
        LOGGER.info(f"Running trainer #{trainer_num}!")

    def on_close(self, *args, **kwargs):
        """
        The user attempted to close the trainer.
        """
        sys.exit(0)

    def run(self):
        """
        Run the trainer.
        """
        self.model.handle_start()
        while True:
            self.model.update(is_training=True)


class TrainerManager:
    """
    A controlling manager class for trainers.
    """

    def __init__(self, *args, **kwargs):
        signal.signal(signal.SIGINT, self.on_close)
        num_trainers = max(1, cpu_count() - 1)
        for trainer_num in range(num_trainers):
            trainer = Process(target=self.manager, args=(trainer_num,) + args, kwargs=kwargs)
            trainer.daemon = True
            trainer.start()

    def manager(self, trainer_num, *args, **kwargs):
        """
        Manager who restarts new trainers when they complete their task.
        """
        while True:
            Trainer(trainer_num, *args, **kwargs).run()

    def on_close(self, *args, **kwargs):
        """
        The user attempted to close the manager.
        """
        sys.exit(0)

    def run(self):
        """
        Run the trainer handler. Essentially sleep and wait for an exit command.
        """
        while True:
            sleep(0.1)

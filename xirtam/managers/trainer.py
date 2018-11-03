"""
Module containing high level class for controlling policy training.
"""
import sys
import signal
import logging
from time import sleep, perf_counter
from multiprocessing import Process, cpu_count
from xirtam.core.model import Model
from xirtam.core.planner import (
    TrainingCompletedException,
    TimeLimitException,
    TrainingQualityException,
)
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
        self.last_tick = perf_counter()
        LOGGER.info(f"Running trainer #{trainer_num} on world {self.model.world.__hash__()}!")

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
            current_tick = perf_counter()
            self.model.update(delta_time=current_tick - self.last_tick, is_training=True)
            self.last_tick = current_tick


class TrainerManager:
    """
    A controlling manager class for trainers.
    """

    def __init__(self, *args, **kwargs):
        signal.signal(signal.SIGINT, self.on_close)
        num_trainers = max(1, cpu_count() - 1)
        for trainer_num in range(num_trainers):
            trainer = Process(target=self.manage, args=(trainer_num,) + args, kwargs=kwargs)
            trainer.daemon = True
            trainer.start()

    def manage(self, trainer_num, *args, **kwargs):
        """
        Restart trainers when they complete their task.
        """
        while True:
            try:
                Trainer(trainer_num, *args, **kwargs).run()
            except (TrainingCompletedException, TimeLimitException, TrainingQualityException):
                continue

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

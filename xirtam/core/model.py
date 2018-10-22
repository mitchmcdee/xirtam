"""
Module containing generic Model related elements. More specific elements (such as
the planner) are further separated into additional modules.
"""
import logging
from xirtam.core.robot import Robot
from xirtam.core.world import World
from xirtam.core.planner import Planner

LOGGER = logging.getLogger(__name__)


class Model:
    """
    The governing class for the simulation model. Represents all the data within the simulation,
    which can be manipulated and interacted with.
    """

    def __init__(self, world_path, robot_path, motion_path, output_path, model_path):
        self.world = World(world_path)
        self.robot = Robot(robot_path)
        self.planner = Planner(self.robot, self.world, motion_path, output_path, model_path)

    def handle_start(self):
        """
        Handle the user attempting to start the planning.
        """
        self.planner.handle_start()

    def handle_reset(self):
        """
        Handle the user attempting to reset the simulation.
        """
        self.world.handle_reset()
        self.planner.handle_reset()

    def handle_pause(self):
        """
        Handle the user attempting to pause the simulation.
        """
        self.planner.handle_pause()

    def handle_plus(self):
        """
        Handle the user attempting to increase the speed of the simulation.
        """
        self.planner.handle_plus()

    def handle_minus(self):
        """
        Handle the user attempting to decrease the speed of the simulation.
        """
        self.planner.handle_minus()

    def handle_toggle_view(self):
        """
        Handle the user attempting to toggle the view of the simulation.
        """
        self.planner.handle_toggle_view()

    def update(self, delta_time, is_training=False):
        """
        Perform an update given the elapsed time.
        """
        self.planner.update(delta_time, is_training)

    def draw(self):
        """
        Draw the model components.
        """
        self.planner.draw()

"""
Module containing high level information for the simulation environment,
such as the GUI window.
"""
import sys
import signal
import pyglet
import logging
from model import Model
from view import View
from controller import Controller
from settings import WINDOW_DIMENSIONS, LOG_LEVEL, FPS_LIMIT, IS_FULLSCREEN
from simulation_parser import SimulationParser


# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Simulator(pyglet.window.Window):
    """
    A simulation window containing the simulation environment.
    """

    def __init__(self, world_filepath, robot_filepath, motion_filepath):
        width, height = WINDOW_DIMENSIONS
        super(Simulator, self).__init__(
            width=width, height=height, resizable=True, fullscreen=IS_FULLSCREEN
        )
        # MVC
        self.model = Model(world_filepath, robot_filepath, motion_filepath)
        self.view = View(self, self.model)
        self.controller = Controller(self.model, self.view)
        # Pyglet
        pyglet.clock.schedule(self.update)
        pyglet.clock.set_fps_limit(FPS_LIMIT)
        signal.signal(signal.SIGINT, self.on_close)

    def run(self):
        """
        Run the simulator.
        """
        pyglet.app.run()

    def on_deactivate(self):
        """
        The window was deactivated.
        """
        self.controller.on_deactivate()

    def on_key_press(self, symbol, modifiers):
        """
        A key on the keyboard was pressed (and held down).
        """
        self.controller.on_key_press(symbol, modifiers)

    def on_key_release(self, symbol, modifiers):
        """
        A key on the keyboard was released.
        """
        self.controller.on_key_release(symbol, modifiers)

    def on_mouse_press(self, x, y, button, modifiers):
        """
        A mouse button was pressed (and held down).
        """
        self.controller.on_mouse_press(x, y, button, modifiers)

    def on_mouse_motion(self, x, y, dx, dy):
        """
        The mouse was moved with no buttons held down.
        """
        self.controller.on_mouse_motion(x, y, dx, dy)

    def on_close(self, *args, **kwargs):
        """
        The user attempted to close the window.
        """
        self.controller.on_close()

    def on_resize(self, width, height):
        """
        The window was resized.
        """
        self.controller.on_resize(width, height)

    def update(self, delta_time):
        """
        Perform an update given the elapsed time step.
        """
        self.controller.update(delta_time)

    def on_draw(self):
        """
        The window contents must be redrawn.
        """
        self.controller.draw()


if __name__ == "__main__":
    parsed_args = SimulationParser().parse_args(sys.argv[1:])
    LOGGER.info("Running simulation...")
    Simulator(parsed_args.world_path, parsed_args.robot_path, parsed_args.motion_path).run()
    LOGGER.info("Simulation finished.")

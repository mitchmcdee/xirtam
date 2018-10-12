"""
Module containing generic Controller related elements. More specific elements (such as
key handling) are further separated into additional modules.
"""
import logging
import pyglet
from xirtam.core.settings import Command
from xirtam.utils.key_handler import KeyHandler, DebouncedKeyHandler

LOGGER = logging.getLogger(__name__)


class Controller:
    """
    The governing class for controling the simulation. Handles and delegates control to
    respective bodies were appropriate.
    """

    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.key_handler = KeyHandler()
        self.debounced_key_handler = DebouncedKeyHandler()
        self.activated = True

    def on_key_press(self, symbol, modifiers):
        """
        A key on the keyboard was pressed (and held down).
        """
        if self.activated:
            self.key_handler.add(symbol)
            self.debounced_key_handler.add(symbol)

    def on_key_release(self, symbol, modifiers):
        """
        A key on the keyboard was released.
        """
        self.key_handler.remove(symbol)
        self.debounced_key_handler.remove(symbol)

    def on_mouse_press(self, x, y, button, modifiers):
        """
        A mouse button was pressed (and held down).
        """
        self.handle_activate()

    def on_mouse_motion(self, x, y, dx, dy):
        """
        The mouse was moved with no buttons held down.
        """
        if self.activated:
            self.view.on_mouse_motion(x, y, dx, dy)

    def on_deactivate(self):
        """
        The window was deactivated.
        """
        self.handle_deactivate()

    def on_close(self):
        """
        The user attempted to close the window.
        """
        self.handle_quit()

    def on_resize(self, width, height):
        """
        The window was resized.
        """
        self.view.on_resize(width, height)

    def handle_activate(self):
        """
        The window was activated.
        """
        self.activated = True
        self.view.handle_activate()

    def handle_deactivate(self):
        """
        The window was deactivated.
        """
        self.activated = False
        self.view.handle_deactivate()

    def handle_quit(self):
        """
        Handle the user attempting to quit the simulation.
        """
        pyglet.app.exit()

    def handle_escape(self):
        """
        Handle the user attempting to escape the window.
        """
        self.activated = False
        self.view.handle_escape()

    def handle_keys(self, delta_time):
        """
        Handle the keys currently pressed and execute any commands.
        """
        if Command.ESCAPE in self.debounced_key_handler:
            self.handle_escape()

        if Command.QUIT in self.debounced_key_handler:
            self.handle_quit()

        if Command.PAUSE in self.debounced_key_handler:
            self.model.handle_pause()

        if Command.START in self.debounced_key_handler:
            self.model.handle_start()

        if Command.RESET in self.debounced_key_handler:
            self.model.handle_reset()

        if Command.TOGGLE_WORLD in self.debounced_key_handler:
            self.model.handle_toggle_world()

        if Command.LEFT in self.key_handler:
            self.view.handle_left(delta_time)

        if Command.RIGHT in self.key_handler:
            self.view.handle_right(delta_time)

        if Command.FORWARD in self.key_handler:
            self.view.handle_forward(delta_time)

        if Command.BACKWARD in self.key_handler:
            self.view.handle_backward(delta_time)

        if Command.DOWN in self.key_handler:
            self.view.handle_down(delta_time)

        if Command.UP in self.key_handler:
            self.view.handle_up(delta_time)

    def update(self, delta_time):
        """
        Perform an update given the elapsed time step.
        """
        self.handle_keys(delta_time)
        self.model.update(delta_time)

    def draw(self):
        """
        The window contents must be redrawn.
        """
        self.view.draw()

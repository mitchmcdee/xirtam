"""
Module containing generic View related elements. More specific elements (such as
camera) are further separated into additional modules.
"""
import pyglet
import logging
from pyglet.gl import (
    glEnable,
    GL_DEPTH_TEST,
    GL_PROJECTION,
    GL_MODELVIEW,
    gluPerspective,
    glLoadIdentity,
    glMatrixMode,
    GL_SMOOTH,
    glShadeModel,
)
from xirtam.core.camera import FirstPersonCamera
from xirtam.core.settings import (
    FOV,
    NEAR_PLANE,
    FAR_PLANE,
    SHOW_FPS,
    IS_FULLSCREEN,
    WINDOW_DIMENSIONS,
)

LOGGER = logging.getLogger(__name__)


class View:
    """
    The governing class for simulation visualisation as well as interaction.
    """

    def __init__(self, window, model, show_fps=True):
        self.window = window
        self.model = model
        self.camera = FirstPersonCamera()
        if SHOW_FPS:
            self.fps_display = pyglet.window.FPSDisplay(self.window)
        # OpenGL init
        self.on_resize(self.window.width, self.window.height)
        self.window.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        """
        The mouse was moved with no buttons held down.
        """
        self.camera.on_mouse_motion(x, y, dx, dy)

    def on_resize(self, width, height):
        """
        The window was resized.
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, width / height, NEAR_PLANE, FAR_PLANE)
        glMatrixMode(GL_MODELVIEW)
        glShadeModel(GL_SMOOTH)

    def handle_left(self, delta_time):
        """
        Handle the user attempting to go left.
        """
        self.camera.handle_left(delta_time)

    def handle_right(self, delta_time):
        """
        Handle the user attempting to go right.
        """
        self.camera.handle_right(delta_time)

    def handle_forward(self, delta_time):
        """
        Handle the user attempting to go forward.
        """
        self.camera.handle_forward(delta_time)

    def handle_backward(self, delta_time):
        """
        Handle the user attempting to go backward.
        """
        self.camera.handle_backward(delta_time)

    def handle_down(self, delta_time):
        """
        Handle the user attempting to go down.
        """
        self.camera.handle_down(delta_time)

    def handle_up(self, delta_time):
        """
        Handle the user attempting to go up.
        """
        self.camera.handle_up(delta_time)

    def handle_escape(self):
        """
        Handle the user attempting to escape the window.
        """
        self.handle_deactivate()

    def handle_activate(self):
        """
        The window was activated.
        """
        width, height = WINDOW_DIMENSIONS
        self.window.set_fullscreen(IS_FULLSCREEN, width=width, height=height)
        self.window.set_exclusive_mouse(True)

    def handle_deactivate(self):
        """
        The window was deactivated.
        """
        self.window.set_fullscreen(False)
        self.window.set_exclusive_mouse(False)

    def draw(self):
        """
        The window contents must be redrawn.
        """
        self.window.clear()
        # Draw closest first
        glEnable(GL_DEPTH_TEST)
        self.model.draw()
        self.camera.draw()
        if SHOW_FPS:
            self.fps_display.draw()

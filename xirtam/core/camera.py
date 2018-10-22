"""
Camera object for scene viewing and navigating.
"""
import logging
import math
import pyglet
from xirtam.core.world import World
from xirtam.core.settings import MOUSE_SENSITIVITY, MOVEMENT_SPEED, INVERTED_Y

LOGGER = logging.getLogger(__name__)


class FirstPersonCamera:
    """
    A first person camera view of the scene.
    """

    def __init__(self, world: World):
        # Look directly down onto world.
        self.dx = world.x + world.width / 2
        self.dy = world.y + world.height / 2
        self.dz = max(world.height, world.width) * 1.5
        self.pitch = 0.0
        self.yaw = 270.0

    def on_mouse_motion(self, x, y, dx, dy):
        """
        The mouse was moved with no buttons held down.
        """
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch += dy * MOUSE_SENSITIVITY * (-1 if INVERTED_Y else 1)

    def handle_left(self, delta_time):
        """
        Handle the user attempting to go left.
        """
        distance_travelled = delta_time * MOVEMENT_SPEED
        yaw_deg = math.radians(self.yaw)
        self.dx -= distance_travelled * math.cos(yaw_deg)
        self.dy += distance_travelled * math.sin(yaw_deg)

    def handle_right(self, delta_time):
        """
        Handle the user attempting to go right.
        """
        distance_travelled = delta_time * MOVEMENT_SPEED
        yaw_deg = math.radians(self.yaw)
        self.dx += distance_travelled * math.cos(yaw_deg)
        self.dy -= distance_travelled * math.sin(yaw_deg)

    def handle_down(self, delta_time):
        """
        Handle the user attempting to go down.
        """
        self.dz -= delta_time * MOVEMENT_SPEED

    def handle_up(self, delta_time):
        """
        Handle the user attempting to go up.
        """
        self.dz += delta_time * MOVEMENT_SPEED

    def handle_forward(self, delta_time):
        """
        Handle the user attempting to go forward.
        """
        distance_travelled = delta_time * MOVEMENT_SPEED
        yaw_deg = math.radians(self.yaw)
        pitch_deg = math.radians(self.pitch)
        self.dx -= distance_travelled * math.sin(yaw_deg) * math.sin(pitch_deg)
        self.dy -= distance_travelled * math.cos(yaw_deg) * math.sin(pitch_deg)
        self.dz -= distance_travelled * math.cos(pitch_deg)

    def handle_backward(self, delta_time):
        """
        Handle the user attempting to go backward.
        """
        distance_travelled = delta_time * MOVEMENT_SPEED
        yaw_deg = math.radians(self.yaw)
        pitch_deg = math.radians(self.pitch)
        self.dx += distance_travelled * math.sin(yaw_deg) * math.sin(pitch_deg)
        self.dy += distance_travelled * math.cos(yaw_deg) * math.sin(pitch_deg)
        self.dz += distance_travelled * math.cos(pitch_deg)

    def draw(self):
        """
        Apply camera transformations.
        """
        pyglet.gl.glLoadIdentity()
        pyglet.gl.glRotatef(self.pitch, 1.0, 0.0, 0.0)
        pyglet.gl.glRotatef(self.yaw, 0.0, 0.0, 1.0)
        pyglet.gl.glTranslatef(-self.dx, -self.dy, -self.dz)

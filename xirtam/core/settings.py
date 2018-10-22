"""
General and specific settings for running and controlling the simulation environment.
"""
import pyglet
from enum import Enum


# Command
class Command(Enum):
    """
    Simulation commands with the input keys needed to perform them.
    """

    ESCAPE = [pyglet.window.key.ESCAPE]
    QUIT = [pyglet.window.key.Q]
    FORWARD = [pyglet.window.key.W]
    BACKWARD = [pyglet.window.key.S]
    LEFT = [pyglet.window.key.A]
    RIGHT = [pyglet.window.key.D]
    UP = [pyglet.window.key.SPACE]
    DOWN = [pyglet.window.key.LSHIFT]
    START = [pyglet.window.key.ENTER]
    PAUSE = [pyglet.window.key.P]
    RESET = [pyglet.window.key.R]
    TOGGLE_VIEW = [pyglet.window.key.T]
    PLUS = [pyglet.window.key.EQUAL]
    MINUS = [pyglet.window.key.MINUS]
    # SAVE = [pyglet.window.key.LCOMMAND, pyglet.window.key.S]
    # OPEN = [pyglet.window.key.LCOMMAND, pyglet.window.key.O]


# Logging
LOG_LEVEL = "INFO"

# Camera
MOVEMENT_SPEED = 3
MOUSE_SENSITIVITY = 0.1
INVERTED_Y = True

# Window
WINDOW_DIMENSIONS = (1440, 900)
ASPECT_RATIO = WINDOW_DIMENSIONS[0] / WINDOW_DIMENSIONS[1]
FPS_LIMIT = 60.0
FOV = 45.0
NEAR_PLANE = 0.01
FAR_PLANE = 1000.0
SHOW_FPS = True
IS_FULLSCREEN = False

# Robot rendering
NUM_FOOT_POINTS = 25
START_COLOUR = (15, 75, 75)
GOAL_COLOUR = (15, 15, 75)
PLANNING_COLOUR = (15, 75, 15)
EXECUTING_COLOUR = (75, 75, 15)
BODY_ALT_MODIFIER = 4.0
ROBOT_LINE_WIDTH = 5.0
EXECUTION_FPS_LIMIT = 200.0
FPS_JUMP = EXECUTION_FPS_LIMIT / 20

# World rendering
VALID_COLOUR = (50, 200, 50)
INVALID_COLOUR = (200, 50, 50)
MAX_PERMEABILITY_COLOUR = 50
MIN_PERMEABILITY_COLOUR = 200

# CNN
OUTPUT_BMP_DIMENSIONS = (128, 128)
OUTPUT_VALID_COLOUR = 0
OUTPUT_DEFAULT_COLOUR = 55
OUTPUT_INVALID_COLOUR = 255
BELIEF_DIMENSIONS = (16, 16)

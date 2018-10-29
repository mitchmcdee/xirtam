"""
General and specific settings for running and controlling the simulation environment.
"""
import logging
import pyglet
from math import inf
from enum import Enum


# Command
class Command(Enum):
    """
    Simulation commands with the input keys needed to perform them.
    """

    # Escape the simulation (i.e. unfocus the window)
    ESCAPE = [pyglet.window.key.ESCAPE]
    # Quit the simulation.
    QUIT = [pyglet.window.key.Q]
    # Move camera forward.
    FORWARD = [pyglet.window.key.W]
    # Move camera backward.
    BACKWARD = [pyglet.window.key.S]
    # Move camera left.
    LEFT = [pyglet.window.key.A]
    # Move camera right.
    RIGHT = [pyglet.window.key.D]
    # Move camera up.
    UP = [pyglet.window.key.SPACE]
    # Move camera down.
    DOWN = [pyglet.window.key.LSHIFT]
    # Start the simulation.
    START = [pyglet.window.key.ENTER]
    # Pause the simulation.
    PAUSE = [pyglet.window.key.P]
    # Reset the simulation.
    RESET = [pyglet.window.key.R]
    # Toggle the underlying world view.
    TOGGLE_VIEW = [pyglet.window.key.T]
    # Increase the execution FPS.
    INCREASE_FPS = [pyglet.window.key.EQUAL]
    # Decrease the execution FPS.
    DECREASE_FPS = [pyglet.window.key.MINUS]


# Logging
# Minimum level to log.
LOG_LEVEL = logging.INFO

# Camera
# Movement speed for camera control.
MOVEMENT_SPEED = 6
# Movement sensitivity for mouse control.
MOUSE_SENSITIVITY = 0.1
# Boolean representing whether to invert the Y direction for mouse control.
INVERTED_Y = True

# Window
# Simulation window dimensions.
WINDOW_DIMENSIONS = (1440, 900)
# Simulation window aspect ratio.
ASPECT_RATIO = WINDOW_DIMENSIONS[0] / WINDOW_DIMENSIONS[1]
# Simulation FPS limit.
FPS_LIMIT = 60.0
# Simulation field of view.
FOV = 45.0
# Minimum distance before clipping occurs.
NEAR_PLANE = 0.01
# Maximum distance before clipping occurs.
FAR_PLANE = 1000.0
# Boolean representing whether to show the FPS display.
SHOW_FPS = True
# Boolean representing whether to fullscreen the simulation window.
IS_FULLSCREEN = False

# Robot rendering
# Number of exterior points for the foot polygon.
NUM_FOOT_POINTS = 25
# RGB Colour for the start configuration.
START_COLOUR = (15, 75, 75)
# RGB Colour for the goal configuration.
GOAL_COLOUR = (15, 15, 75)
# RGB Colour for an executing configuration.
EXECUTING_COLOUR = (75, 75, 15)
# Multiplier for the alternate robot colour, applied to original colour.
ALT_COLOUR_MULTIPLIER = 4.0
# Width of robot lines.
ROBOT_LINE_WIDTH = 5.0
# Maximum limit for execution FPS.
EXECUTION_FPS_LIMIT = 200.0
# Amount user is able to increase/decrease execution FPS by.
FPS_JUMP = EXECUTION_FPS_LIMIT / 20

# World rendering
# RGB Colour for an valid foot placement.
VALID_PLACEMENT_COLOUR = (50, 200, 50)
# RGB Colour for an invalid foot placement.
INVALID_PLACEMENT_COLOUR = (200, 50, 50)
# Greyscale colour for the maximum permeability.
MAX_PERMEABILITY_COLOUR = 50
# Greyscale colour for the minimum permeability.
MIN_PERMEABILITY_COLOUR = 200

# CNN + Belief modelling/sampling
# Width and height for the output bitmap image.
OUTPUT_BMP_DIMENSIONS = (128, 128)
# Output greyscale value for valid foot placements.
OUTPUT_VALID_COLOUR = 0
# Default output greyscale value (i.e. the background).
OUTPUT_DEFAULT_COLOUR = 55
# Output greyscale value for invalid foot placements.
OUTPUT_INVALID_COLOUR = 255

# Planning/Training
# Image output limit before trainer is killed.
OUTPUT_LIMIT = inf
# Time in seconds before trainer is killed.
TIME_LIMIT = 5 * 60
# Width and height of the belief graph.
BELIEF_DIMENSIONS = (64, 64)
# Number of colours to include in the traversable belief graph.
BELIEF_LEVELS = 8
# Dictates the proportion of the world to increase jiggle by every sample attempt.
JIGGLE_FACTOR = 2000
# Number of turning point sample attempts before restarting entire path find.
SAMPLE_LIMIT = 1000

"""
Module containing path planning components.
"""
import os
import csv
import logging
import pyglet
import math
import sys
import networkx as nx
import numpy as np
from enum import Enum
from random import uniform
from itertools import cycle
from scipy.misc import imresize
from collections import defaultdict
from typing import List  # noqa: F401
from xirtam.utils.geometry.point2d import Point2D
from xirtam.utils.geometry.vector2d import Vector2D
from xirtam.core.robot import RobotConfig, Robot
from xirtam.utils.utils import get_coerced_reader_row_helper, translate
from keras.models import load_model
from xirtam.core.settings import (
    START_COLOUR,
    GOAL_COLOUR,
    EXECUTING_COLOUR,
    ROBOT_LINE_WIDTH,
    EXECUTION_FPS_LIMIT,
    BELIEF_DIMENSIONS,
    BELIEF_LEVELS,
    FPS_JUMP,
    JIGGLE_FACTOR,
    SAMPLE_LIMIT,
    OUTPUT_LIMIT,
    TIME_LIMIT,
)

LOGGER = logging.getLogger(__name__)
LEGS = range(Robot.NUM_LEGS)


class OutputLimitException(Exception):
    """
    Raised when a trainer has reached its image output limit.
    """

    pass


class TimeLimitException(Exception):
    """
    Raised when a trainer has elapsed its total allocated run time.
    """

    pass


class TrainingQualityException(Exception):
    """
    Raised when a trainer has determined the current map is of insufficient training quality.
    """

    pass


class ViewState(Enum):
    """
    Visibility state of the simulation.
    """

    REGIONS_PLACEMENTS = 0
    PLACEMENTS = 1
    OFF = 2
    BELIEFS = 3
    BELIEFS_PLACEMENTS = 4
    REGIONS = 5


class Planner:
    """
    A robot path planner that executes paths influenced by a core belief of the environment.
    """

    # Directions connecting cells in a grid.
    # In order: left, right, up, down, bottom-right, bottom-left, top-right, top-left.
    DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    def __init__(self, robot, world, motion_path, output_path, model_path):
        # Time taken during this training run.
        self.total_time = 0
        # Number of images the planner has output.
        self.output_count = 0
        # Execution FPS.
        self.execution_fps = EXECUTION_FPS_LIMIT / 2
        # Amount of time in seconds since the last execution move.
        self.last_execution = 0
        # Boolean representing whether the planner is paused.
        self.is_paused = False
        # Boolean representing whether the planner has started planning.
        self.is_started = False
        # Boolean representing whether the planner is currently executing.
        self.is_executing = False
        # Boolean representing whether the planner has gotten to goal.
        self.is_complete = False
        # Current configuration in the execution sequence.
        self.current_config = None
        # Motion plan start configuration.
        self.start_config = None
        # List of configurations since the last valid "checkpoint" configuration.
        # Note: A checkpoint configuration is one that occurs every NUM_LEGS + body configurations.
        self.previous_configs = []  # type: List["RobotConfig"]
        # View of the current world belief, if any.
        self.belief_view = None
        # Graph containing potentially connected belief regions
        self.belief_graph = nx.DiGraph()
        # Neural belief model.
        self.model = None
        # Number of path execution attempts made during this training run.
        self.num_attempts = 0
        # Number of steps taken during this training run.
        self.num_steps = 0
        # Distance travelled during this training run.
        self.run_distance = 0
        # Amount of time in seconds since the planner has started.
        self.run_time = 0
        self.start_config, self.goal_config = self.get_motion_config(robot, motion_path)
        self.robot = robot
        self.world = world
        # If we have a model path, use a neural belief model, else use an occupancy belief map.
        if model_path is None:
            self.get_belief = self.get_occupancy_belief
        else:
            self.model = load_model(model_path)
            self.get_belief = self.get_neural_belief
        output_directory = f"robot-{self.robot.__hash__()}/world-{self.world.__hash__()}"
        self.output_path = os.path.join(output_path, output_directory)
        self.visibility_iterator = cycle(ViewState)
        self.visibility_state = next(self.visibility_iterator)
        self.initialise()

    def get_motion_config(self, robot, motion_path):
        """
        Returns the motion path configuration of the given motion path.
        """
        with open(motion_path) as motion_file:
            motion_reader = csv.reader(motion_file)
            get_motion_row = get_coerced_reader_row_helper(motion_reader, motion_path)
            position = get_motion_row([float] * 2, "start position")
            heading = math.radians(get_motion_row([float], "start heading"))
            foot_vertices = [get_motion_row([float] * 2, "start foot vertex") for _ in LEGS]
            start_config = RobotConfig(robot, position, heading, foot_vertices, START_COLOUR)
            position = get_motion_row([float] * 2, "goal position")
            heading = math.radians(get_motion_row([float], "goal heading"))
            foot_vertices = [get_motion_row([float] * 2, "goal foot vertex") for _ in LEGS]
            goal_config = RobotConfig(robot, position, heading, foot_vertices, GOAL_COLOUR)
        return start_config, goal_config

    def initialise(self):
        """
        Initialise planner.
        """
        self.execution_fps = EXECUTION_FPS_LIMIT / 2
        self.last_execution = 0
        self.num_attempts = self.run_distance = self.num_steps = self.run_time = 0
        self.is_paused = self.is_started = self.is_executing = self.is_complete = False
        self.current_config = self.start_config
        self.previous_configs = []
        self.belief_view = None
        self.update_current_belief()

    def handle_toggle_view(self):
        """
        Handle the user attempting to toggle the view of the simulation.
        """
        self.visibility_state = next(self.visibility_iterator)
        # If we have no beliefs, skip over those states.
        while self.belief_view is None and "BELIEFS" in self.visibility_state.name:
            self.visibility_state = next(self.visibility_iterator)

    def handle_increase_fps(self):
        """
        Handle the user attempting to increase the speed of the simulation.
        """
        self.execution_fps = min(EXECUTION_FPS_LIMIT, self.execution_fps + FPS_JUMP)

    def handle_decrease_fps(self):
        """
        Handle the user attempting to decrease the speed of the simulation.
        """
        self.execution_fps = max(sys.float_info.epsilon, self.execution_fps - FPS_JUMP)

    def handle_reset(self):
        """
        Handle the user attempting to reset the simulation.
        """
        self.initialise()

    def handle_start(self):
        """
        Handle the user attempting to start the simulation.
        """
        self.is_started = True

    def handle_pause(self):
        """
        Handle the user attempting to pause the simulation.
        """
        if self.is_started:
            self.is_paused = not self.is_paused

    def handle_complete(self, is_training):
        """
        Handle the robot completing a path to the goal.
        """
        # If path was straight to goal (i.e. no planning attempts needed), short circuit training.
        if is_training and self.num_attempts == 0:
            LOGGER.info("Pointless training sample, quitting!")
            raise TrainingQualityException()
        LOGGER.info(
            f"Got to goal! Took {self.num_steps} steps in {self.num_attempts} attempts "
            + f"travelling {self.run_distance:.2f} metres in {self.run_time:.2f} seconds "
            + f"to reach the goal for world {self.world.__hash__()}!"
        )
        self.is_complete = True
        self.update_current_belief()
        if is_training:
            # Save final placements.
            self.world.save_placements_bmp(self.robot, self.output_path)
            self.output_count += 1
            # Restart training immediately.
            self.world.handle_reset()
            self.handle_reset()
            self.handle_start()

    def set_belief_view(self, belief):
        """
        Sets the view of the current world belief.
        """
        flat_belief = [belief_level for row in belief for belief_level in row]
        raw_belief = (pyglet.gl.GLubyte * len(flat_belief))(*flat_belief)
        self.belief_view = pyglet.image.ImageData(*belief.shape, "L", raw_belief)

    def update_current_belief(self, belief=None):
        """
        Updates the current belief of the underlying environment.
        """
        self.belief_graph.clear()
        if belief is None:
            belief = self.get_belief()
        self.set_belief_view(belief)
        # Down sample and quantize image.
        belief = imresize(belief, BELIEF_DIMENSIONS, interp="bilinear")
        level_bins = np.linspace(np.min(belief), np.max(belief), BELIEF_LEVELS)
        belief = np.digitize(belief.flat, level_bins).reshape(belief.shape)
        # Set belief weights.
        for y, row in enumerate(belief):
            for x, belief_level in enumerate(row):
                self.belief_graph.add_node((x, y), weight=belief_level)

    def get_occupancy_belief(self):
        """
        Gets the current belief of the environment from the occupancy map.
        """
        return np.array(self.world.get_placements_bmp(self.robot))

    def get_neural_belief(self):
        """
        Gets the current belief of the environment from the neural model.
        """
        placements = self.world.get_placements_bmp(self.robot)
        placements = np.array(placements).astype(float) / 255
        placements_size = placements.shape
        # Reshape to 4D tensor (sample_size, image_width, image_height, num_channels).
        placements = np.reshape(placements, (1, *placements_size, 1))
        belief = self.model.predict(placements)
        return (belief * 255).astype(int).reshape(*placements_size)

    def update(self, delta_time, is_training):
        """
        Perform an update given the elapsed time.
        """
        self.total_time += delta_time
        self.run_time += delta_time
        if is_training and self.total_time >= TIME_LIMIT:
            LOGGER.info("Reached time limit, quitting!")
            raise TimeLimitException()
        if is_training and self.output_count >= OUTPUT_LIMIT:
            LOGGER.info("Reached output limit, quitting!")
            raise OutputLimitException()
        if self.is_complete or self.is_paused or not self.is_started:
            return
        if self.is_executing:
            self.execute(delta_time, is_training)
        else:
            self.plan()

    def execute(self, delta_time, is_training):
        """
        Perform a single iteration of execution.
        """
        self.last_execution += delta_time

        # If we're not training and execution time hasn't been reached, return for now.
        if not is_training and self.last_execution < (1 / self.execution_fps):
            return
        self.last_execution = 0

        # If we're at goal, completed.
        if self.current_config == self.goal_config:
            self.handle_complete(is_training)
            return

        # Sample the current config in the world for validity (i.e. check all feet can adhere).
        if self.world.is_valid_config(self.current_config):
            # Only keep past Robot.NUM_LEGS + BODY configurations for checkpointing.
            if len(self.previous_configs) == Robot.NUM_LEGS + 1:
                self.previous_configs.clear()
            self.previous_configs.append(self.current_config.copy())
        else:
            self.update_current_belief()
            LOGGER.debug("Detected invalid region! Back-tracking now.")
            self.execution_path = self.get_path_to_previous()
            if is_training:
                self.world.save_placements_bmp(self.robot, self.output_path)
                self.output_count += 1

        # Get next execution step.
        next_config = next(self.execution_path, None)

        # If we've exhausted the execution path, start planning again.
        if next_config is None:
            self.previous_configs.clear()
            self.is_executing = False
            self.num_attempts += 1
            return

        # Take next step.
        self.run_distance += self.current_config.distance_to(next_config)
        self.current_config = next_config
        self.num_steps += 1

    def get_config_edges(self, config_a, config_b):
        """
        Returns the edges connecting config a and b.
        """
        config_edges = []
        interpolations = config_a.interpolate(config_b, self.world)
        if interpolations is not None:
            config_edges.append((config_a, config_b))
        interpolations = config_b.interpolate(config_a, self.world)
        if interpolations is not None:
            config_edges.append((config_b, config_a))
        return config_edges

    def get_belief_path(self):
        """
        Returns a path to the goal config, influenced by the belief of the underlying environment.
        """
        # Separate cells by their levels.
        level_cells = defaultdict(list)
        for cell, cell_data in self.belief_graph.nodes(data=True):
            belief_level = cell_data["weight"]
            level_cells[belief_level].append(cell)

        # Determine current and goal cells in belief graph.
        belief_width, belief_height = BELIEF_DIMENSIONS
        world_left, world_top, world_right, world_bottom = self.world.bounds
        start_x = int(translate(self.current_config.x, world_left, world_right, 0, belief_width))
        start_y = int(translate(self.current_config.y, world_bottom, world_top, 0, belief_height))
        start_cell = (start_x, start_y)
        goal_x = int(translate(self.goal_config.x, world_left, world_right, 0, belief_width))
        goal_y = int(translate(self.goal_config.y, world_bottom, world_top, 0, belief_height))
        goal_cell = (goal_x, goal_y)

        # Incrementally add cell edges to graph. We do this so that we can take the highest belief
        # path possible before attempting to navigate through "walls" of the belief model.
        for belief_level, cells in level_cells.items():
            for cell in cells:
                cell_x, cell_y = cell
                for delta_x, delta_y in self.DIRECTIONS:
                    neighbour_x, neighbour_y = cell_x + delta_x, cell_y + delta_y
                    if neighbour_x in (-1, belief_width) or neighbour_y in (-1, belief_height):
                        continue
                    neighbour = (neighbour_x, neighbour_y)
                    self.belief_graph.add_edge(neighbour, cell)
            if nx.has_path(self.belief_graph, start_cell, goal_cell):
                break

        def get_euclidean_distance(a, b):
            """
            Returns the euclidean distance between cells a and b.
            """
            return Point2D(*a).distance_to(Point2D(*b))

        # Obtain cell path and determine where turning points occur.
        cell_path = nx.astar_path(self.belief_graph, start_cell, goal_cell, get_euclidean_distance)
        previous_direction, previous_cell = None, cell_path[0]
        turning_points = []
        for cell in cell_path[1:]:
            direction = (cell[0] - previous_cell[0], cell[1] - previous_cell[1])
            # If directions change, we've found a turning point.
            if direction != previous_direction:
                turning_points.append((previous_cell, direction))
            previous_direction, previous_cell = direction, cell
        turning_points.append((previous_cell, previous_direction))

        JIGGLE_LIMIT = max(self.world.width, self.world.height)
        JIGGLE_JUMP = JIGGLE_LIMIT / JIGGLE_FACTOR
        # Sample at turning points and add their connected interpolations to path.
        previous_sample = self.current_config
        interpolated_path = []
        for i, ((point_x, point_y), direction) in enumerate(turning_points):
            # Try to sample at the turning point. For every fail, increase the amount of random
            # jiggle to try and find a valid position. This is to avoid trying to place at
            # invalid positions and being stuck in a loop. Don't jiggle more than the world.
            jiggle = 0.0
            for sample_attempt in range(SAMPLE_LIMIT):
                # print(i, jiggle)
                jiggle = min(jiggle + JIGGLE_JUMP, JIGGLE_LIMIT)
                sample_x = translate(point_x, 0, belief_width, world_left, world_right)
                sample_y = translate(point_y, 0, belief_height, world_bottom, world_top)
                offset_x = uniform(-jiggle, jiggle)
                offset_y = uniform(-jiggle, jiggle)
                offset_theta = uniform(-jiggle, jiggle)
                sample_point = Point2D(sample_x + offset_x, sample_y + offset_y)
                sample_heading = offset_theta
                if direction is not None:
                    sample_heading = Vector2D(*direction).angle + offset_theta
                sample = self.robot.get_random_config(self.world, sample_point, sample_heading)
                interpolations = previous_sample.interpolate(sample, self.world)
                # If invalid sample or failed to interpolate, continue sampling.
                if interpolations is None or not sample.is_valid(self.world):
                    continue
                # If we're the final turning point, ensure we can also interpolate to goal.
                if i == len(turning_points) - 1:
                    goal_interpolations = sample.interpolate(self.goal_config, self.world)
                    if goal_interpolations is None:
                        continue
                # Valid sample, break.
                break
            # Exceeded sample limit, exit.
            else:
                LOGGER.debug("Got stuck during belief sampling!")
                return None
            interpolated_path.extend(interpolations)
            previous_sample = sample
        interpolated_path.extend(goal_interpolations)
        return iter(interpolated_path)

    def plan(self):
        """
        Perform a single iteration of planning.
        """
        # If first attempt, try to navigate straight to goal. Else, use current belief.
        if self.num_attempts == 0:
            execution_path = self.get_direct_path_to_goal()
        else:
            execution_path = self.get_belief_path()
        if execution_path is None:
            return
        self.execution_path = execution_path
        self.is_executing = True

    def get_path_to_previous(self):
        """
        Returns the interpolated path to the previous valid config.
        """
        return reversed([c for c in self.previous_configs])

    def get_direct_path_to_goal(self):
        """
        Returns the interpolated path to the previous valid config.
        """
        return iter(self.current_config.interpolate(self.goal_config, self.world))

    def draw(self):
        """
        Draw the world, current motion path, the start and goal configurations as well as any
        generated configurations if planning.
        """
        # Draw the current world view.
        if "REGIONS" in self.visibility_state.name:
            self.world.region_batch.draw()
        if "PLACEMENTS" in self.visibility_state.name:
            self.world.placements_batch.draw()
        if self.belief_view is not None and "BELIEFS" in self.visibility_state.name:
            x, y, width, height = self.world.x, self.world.y, self.world.width, self.world.height
            self.belief_view.blit(x=x, y=y, width=width, height=height)
        # Draw configurations.
        config_batch = pyglet.graphics.Batch()
        pyglet.gl.glLineWidth(ROBOT_LINE_WIDTH)
        self.start_config.draw(config_batch)
        self.goal_config.draw(config_batch)
        if self.current_config not in (None, self.start_config, self.goal_config):
            self.current_config.set_colour(EXECUTING_COLOUR)
            self.current_config.draw(config_batch)
        config_batch.draw()

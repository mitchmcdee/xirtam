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
from functools import partial
from typing import List  # noqa: F401
from xirtam.utils.geometry.point2d import Point2D
from xirtam.utils.geometry.vector2d import Vector2D
from xirtam.core.robot import RobotConfig, Robot
from xirtam.utils.utils import get_coerced_reader_row_helper, translate
from xirtam.neural.timtamnet import TimTamNet
from xirtam.core.settings import (
    START_COLOUR,
    GOAL_COLOUR,
    EXECUTING_COLOUR,
    PLANNING_COLOUR,
    ROBOT_LINE_WIDTH,
    EXECUTION_FPS_LIMIT,
    OUTPUT_BMP_DIMENSIONS,
    BELIEF_DIMENSIONS,
    BELIEF_LEVELS,
    FPS_JUMP,
    JIGGLE_JUMP,
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
    A robot path planner.
    """

    # Constants
    # Graph size limit before it is reset.
    GRAPH_SIZE_LIMIT = 10
    # Image output limit before trainer is killed.
    OUTPUT_LIMIT = 50
    # Time before trainer is killed.
    TIME_LIMIT = 5 * 60
    # Directions connecting cells in a grid.
    # In order: left, right, up, down, bottom-right, bottom-left, top-right, top-left.
    DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    # Variables
    # Graph containing potentially connected configuration samples.
    graph = nx.DiGraph()
    # Graph containing potentially connected belief regions.
    belief_graph = nx.DiGraph()
    # Amount of time in seconds since the planner has started.
    run_time = 0
    # Number of images the planner has output.
    output_count = 0
    # Belief model for the robot.
    model = None
    # Execution fps limit.
    fps_limit = EXECUTION_FPS_LIMIT / 2
    # Amount of time in seconds since the last execution move.
    time_since_last_execute = 0
    # Boolean representing whether the planner is paused.
    is_paused = False
    # Boolean representing whether the planner has started planning.
    is_started = False
    # Boolean representing whether the planner is currently executing.
    is_executing = False
    # Boolean representing whether the planner has gotten to goal.
    is_complete = False
    # Current configuration in the execution sequence.
    current_config = None
    # Motion plan start configuration.
    start_config = None
    # Last configuration sampled.
    last_sampled_config = None
    # List of configurations since the last valid "checkpoint" configuration.
    # Note: A checkpoint configuration is one that occurs every NUM_LEGS + body configurations.
    previous_configs = []  # type: List["RobotConfig"]
    # View of the current world belief, if any.
    belief_view = None

    def __init__(self, robot, world, motion_path, output_path, model_path):
        self.start_config, self.goal_config = self.get_motion_config(robot, motion_path)
        self.robot = robot
        self.world = world
        self.model = None
        # If we have a model, get samples from the model. Else, get random configs.
        if model_path is None:
            self.sample = self.get_random_sample
        else:
            self.model = TimTamNet(input_shape=(*OUTPUT_BMP_DIMENSIONS, 1))
            self.model.load_weights(model_path)
            self.sample = self.get_model_sample
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
        self.fps_limit = EXECUTION_FPS_LIMIT / 2
        self.time_since_last_execute = 0
        self.is_paused = self.is_started = self.is_executing = self.is_complete = False
        self.current_config = self.start_config
        self.reset_graph()
        self.previous_configs = []
        self.belief_view = None
        self.last_sampled_config = None
        if self.model is not None:
            self.update_current_belief()

    def reset_graph(self):
        """
        Resets the planner graph to its default state.
        """
        self.graph.clear()
        self.graph.add_node(self.current_config)
        self.graph.add_node(self.goal_config)
        # Attempt to add direct path from current to goal.
        self.graph.add_edges_from(self.get_config_edges(self.current_config, self.goal_config))

    def handle_toggle_view(self):
        """
        Handle the user attempting to toggle the view of the simulation.
        """
        self.visibility_state = next(self.visibility_iterator)
        # If we have no beliefs, skip over those states.
        while self.belief_view is None and "BELIEFS" in self.visibility_state.name:
            self.visibility_state = next(self.visibility_iterator)

    def handle_plus(self):
        """
        Handle the user attempting to increase the speed of the simulation.
        """
        self.fps_limit = min(EXECUTION_FPS_LIMIT, self.fps_limit + FPS_JUMP)

    def handle_minus(self):
        """
        Handle the user attempting to decrease the speed of the simulation.
        """
        self.fps_limit = max(sys.float_info.epsilon, self.fps_limit - FPS_JUMP)

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
            belief = self.get_current_belief()
        self.set_belief_view(belief)
        # Down sample image.
        belief = imresize(belief, BELIEF_DIMENSIONS, interp="nearest").flatten()
        # Reduce colour depth.
        reduce_depth = partial(
            translate,
            left_min=np.min(belief),
            left_max=np.max(belief),
            right_min=0,
            right_max=BELIEF_LEVELS - 1,
        )
        belief = np.array(list(map(reduce_depth, belief))).astype(int).reshape(BELIEF_DIMENSIONS)
        # Set belief weights.
        for y, row in enumerate(belief):
            for x, belief_level in enumerate(row):
                self.belief_graph.add_node((x, y), weight=belief_level)

    def get_current_belief(self):
        """
        Gets the current belief of the environment from the model.
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
        self.run_time += delta_time
        if is_training and self.run_time >= self.TIME_LIMIT:
            LOGGER.info("Reached time limit, quitting!")
            raise TimeLimitException()
        if is_training and self.output_count >= self.OUTPUT_LIMIT:
            LOGGER.info("Reached output limit, quitting!")
            raise OutputLimitException()
        if self.is_complete:
            if not is_training:
                return
            # Restart training.
            self.world.handle_reset()
            self.handle_reset()
            self.handle_start()
        if self.is_paused:
            return
        if not self.is_started:
            return
        if self.is_executing:
            self.execute(delta_time, is_training)
        else:
            self.plan(is_training)

    def execute(self, delta_time, is_training):
        """
        Perform a single iteration of execution.
        """
        self.time_since_last_execute += delta_time
        # If we're not training and execution time hasn't been reached, return for now.
        if not is_training and self.time_since_last_execute < (1 / self.fps_limit):
            return
        self.time_since_last_execute = 0
        # If we're at goal, completed.
        if self.current_config == self.goal_config:
            LOGGER.info("Got to goal!")
            self.is_complete = True
            if self.model is not None:
                self.update_current_belief()
            if is_training:
                self.world.save_placements_bmp(self.robot, self.output_path)
                self.output_count += 1
            return
        # Sample the current config in the world for validity (i.e. check footpads adhere).
        if self.world.is_valid_config(self.current_config):
            # Only keep past Robot.NUM_LEGS + BODY configurations for checkpointing.
            if len(self.previous_configs) == Robot.NUM_LEGS + 1:
                self.previous_configs.clear()
            self.previous_configs.append(self.current_config.copy())
        else:
            LOGGER.info("Detected invalid region! Back-tracking now.")
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
            return
        # Take next step.
        self.current_config = next_config
        self.graph.add_node(self.current_config)

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

    def get_random_sample(self):
        """
        Samples new random config and attempts to add it to the graph.
        """
        if len(self.graph.nodes) >= self.GRAPH_SIZE_LIMIT:
            self.reset_graph()
        sample = self.robot.get_random_config(self.world)
        if not sample.is_valid(self.world):
            return
        self.last_sampled_config = sample
        new_config_edges = []
        # Attempt to add sample to nodes in graph.
        for config in self.graph.nodes:
            new_config_edges.extend(self.get_config_edges(sample, config))
        self.graph.add_edges_from(new_config_edges)

    def get_model_sample(self):
        """
        Gets a random sample from the belief model.
        """
        belief = self.get_current_belief()
        self.update_current_belief(belief)
        # Separate cells by their levels.
        level_cells = {level: [] for level in range(BELIEF_LEVELS)}
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
        # Obtain cell path and determine where turning points occur.
        cell_path = nx.dijkstra_path(self.belief_graph, start_cell, goal_cell)
        previous_direction = None
        previous_cell = cell_path[0]
        turning_points = []
        for cell in cell_path[1:]:
            direction = (cell[0] - previous_cell[0], cell[1] - previous_cell[1])
            # If directions changing, turning point.
            if direction != previous_direction:
                turning_points.append((previous_cell, direction))
            previous_direction = direction
            previous_cell = cell
        # Start with clean slate.
        self.reset_graph()
        # Sample at turning points and to graph in their connected path order.
        previous_sample = self.current_config
        for (point_x, point_y), direction in turning_points:
            # Try to sample at the turning point. For every fail, increase the amount of random
            # jiggle to try and find a valid position. This is to avoid trying to place at
            # invalid positions and being stuck in a loop.
            jiggle_amount = 0.0
            while True:
                sample_x = translate(point_x, 0, belief_width, world_left, world_right)
                sample_y = translate(point_y, 0, belief_height, world_bottom, world_top)
                offset_x = uniform(-jiggle_amount, jiggle_amount)
                offset_y = uniform(-jiggle_amount, jiggle_amount)
                offset_theta = uniform(-jiggle_amount, jiggle_amount)
                sample_position = Point2D(sample_x + offset_x, sample_y + offset_y)
                sample_heading = Vector2D(*direction).angle + offset_theta
                sample = self.robot.get_random_config(self.world, sample_position, sample_heading)
                interpolations = previous_sample.interpolate(sample, self.world)
                if interpolations is not None and sample.is_valid(self.world):
                    break
                jiggle_amount += JIGGLE_JUMP
            self.graph.add_edge(previous_sample, sample)
            previous_sample = sample
        # Add final edge to goal.
        self.graph.add_edge(previous_sample, self.goal_config)

    def plan(self, is_training):
        """
        Perform a single iteration of planning.
        """
        # Check if there's a path through configurations starting from current to goal.
        if not nx.has_path(self.graph, self.current_config, self.goal_config):
            self.sample()
            return
        node_path = nx.shortest_path(self.graph, self.current_config, self.goal_config)
        execution_path = self.get_interpolated_path(node_path)
        if execution_path is None:
            return
        # If training and the path is just from start to goal, short circuit training.
        if is_training and self.current_config == self.start_config and len(node_path) == 2:
            LOGGER.info("Pointless training sample, quitting!")
            raise TrainingQualityException()
        LOGGER.info("Found a path to the goal! Executing now.")
        self.execution_path = execution_path
        self.is_executing = True

    def get_path_to_previous(self):
        """
        Returns the interpolated path to the previous valid config.
        """
        return reversed([c for c in self.previous_configs])

    def get_interpolated_path(self, node_path):
        """
        Returns the interpolated node path.
        """
        interpolated_path = []
        previous_node = node_path[0]
        for node in node_path[1:]:
            interpolation = previous_node.interpolate(node, self.world)
            if interpolation is None:
                # Remove bad edge from graph.
                self.graph.remove_edge(previous_node, node)
                return None
            interpolated_path.extend(interpolation)
            previous_node = node
        return iter(interpolated_path)

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
        self.start_config.draw(config_batch)
        self.goal_config.draw(config_batch)
        if self.last_sampled_config is not None:
            self.last_sampled_config.set_colour(PLANNING_COLOUR)
            self.last_sampled_config.draw(config_batch)
        if self.current_config not in (None, self.start_config, self.goal_config):
            self.current_config.set_colour(EXECUTING_COLOUR)
            self.current_config.draw(config_batch)
        pyglet.gl.glLineWidth(ROBOT_LINE_WIDTH)
        config_batch.draw()

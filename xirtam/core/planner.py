"""
Module containing path planning components.
"""
import os
import csv
import logging
import pyglet
import math
import networkx as nx
from xirtam.core.robot import RobotConfig, Robot
from xirtam.utils.utils import get_coerced_reader_row_helper
from xirtam.core.settings import (
    START_COLOUR,
    GOAL_COLOUR,
    EXECUTING_COLOUR,
    PLANNING_COLOUR,
    ROBOT_LINE_WIDTH,
    EXECUTION_FPS_LIMIT,
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


class Planner:
    """
    A robot path planner. Utilises PRM during training and samples from a precomputed
    model during playback simulation.
    """

    # TODO(mitch): tinker with this value?
    # Note: low is good!! might have to change for more complex graphs though
    GRAPH_SIZE_LIMIT = 10
    # TODO(mitch): tinker with this value?
    OUTPUT_LIMIT = 50
    # TODO(mitch): tinker with this value?
    TIME_LIMIT = 5 * 60

    def __init__(self, robot, world, motion_filepath, output_filepath):
        with open(motion_filepath) as motion_file:
            motion_reader = csv.reader(motion_file)
            get_motion_row = get_coerced_reader_row_helper(motion_reader, motion_filepath)
            position = get_motion_row([float] * 2, "start position")
            heading = math.radians(get_motion_row([float], "start heading"))
            foot_vertices = [get_motion_row([float] * 2, "start foot vertex") for _ in LEGS]
            self.start_config = RobotConfig(robot, position, heading, foot_vertices, START_COLOUR)
            position = get_motion_row([float] * 2, "goal position")
            heading = math.radians(get_motion_row([float], "goal heading"))
            foot_vertices = [get_motion_row([float] * 2, "goal foot vertex") for _ in LEGS]
            self.goal_config = RobotConfig(robot, position, heading, foot_vertices, GOAL_COLOUR)
        self.robot = robot
        self.world = world
        robot_hash = self.robot.__hash__()
        world_hash = self.world.__hash__()
        output_directory = f"robot-{robot_hash}/world-{world_hash}"
        self.output_filepath = os.path.join(output_filepath, output_directory)
        self.graph = nx.DiGraph()
        self.output_count = 0
        self.run_time = 0
        self.initialise()

    def initialise(self):
        """
        Initialise planner.
        """
        self.time_since_last_execute = 0
        self.is_paused = False
        self.is_started = False
        self.is_executing = False
        self.is_complete = False
        self.current_config = self.start_config
        self.reset_graph()
        self.previous_configs = []
        self.last_sampled_config = None

    def reset_graph(self):
        """
        Resets the planner graph to its default state.
        """
        self.graph.clear()
        self.graph.add_node(self.current_config)
        self.graph.add_node(self.goal_config)
        self.graph.add_edges_from(self.get_config_edges(self.current_config, self.goal_config))

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
            # Restart training
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
        if not is_training and self.time_since_last_execute < (1 / EXECUTION_FPS_LIMIT):
            return
        self.time_since_last_execute = 0
        # If we're at goal, completed.
        if self.current_config == self.goal_config:
            LOGGER.info("Got to goal!")
            self.is_complete = True
            self.world.save_placements_bmp(self.robot, self.output_filepath)
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
            self.world.save_placements_bmp(self.robot, self.output_filepath)
            self.output_count += 1
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

    def sample(self):
        """
        Samples new config and attempts to add it to the graph.
        """
        # TODO(mitch): sample from precomputed model during playback, sample from config space
        # during training.
        sample = self.robot.get_random_config(self.world)
        if not sample.is_valid(self.world):
            return
        self.last_sampled_config = sample
        new_config_edges = []
        # Attempt to add sample to nodes in graph.
        for config in self.graph.nodes:
            new_config_edges.extend(self.get_config_edges(sample, config))
        self.graph.add_edges_from(new_config_edges)

    def plan(self, is_training):
        """
        Perform a single iteration of planning.
        """
        if len(self.graph.nodes) >= self.GRAPH_SIZE_LIMIT:
            self.reset_graph()
        self.sample()
        # Check if there's a path through configurations starting from current to goal.
        if not nx.has_path(self.graph, self.current_config, self.goal_config):
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
        for i in range(1, len(node_path)):
            previous, current = node_path[i - 1], node_path[i]
            interpolation = previous.interpolate(current, self.world)
            if interpolation is None:
                return None
            interpolated_path.extend(interpolation)
        return iter(interpolated_path)

    def draw(self):
        """
        Draw the current motion path, the start and goal configurations as well as any
        generated configurations if planning.
        """
        batch = pyglet.graphics.Batch()
        self.start_config.draw(batch)
        self.goal_config.draw(batch)
        if self.last_sampled_config is not None:
            self.last_sampled_config.colour = PLANNING_COLOUR
            self.last_sampled_config.draw(batch)
        if all(self.current_config != c for c in (None, self.start_config, self.goal_config)):
            self.current_config.colour = EXECUTING_COLOUR
            self.current_config.draw(batch)
        pyglet.gl.glLineWidth(ROBOT_LINE_WIDTH)
        batch.draw()

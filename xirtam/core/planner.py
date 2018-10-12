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


class Planner:
    """
    A robot path planner. Utilises PRM during training and samples from a precomputed
    model during playback simulation.
    """

    # TODO(mitch): tinker with this value? low is good!!
    GRAPH_SIZE_LIMIT = 3

    def __init__(self, robot, world, motion_filepath, output_path):
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
        world_robot_dir = f'world-{self.world.__hash__()}/robot-{self.robot.__hash__()}'
        self.output_path = os.path.join(output_path, world_robot_dir)
        # Make ouput world-robot directory if it doesn't already exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # Save valid regions bmp for the provided robot.
        self.world.save_regions_bmp(self.robot, self.output_path)
        self.graph = nx.DiGraph()
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
        self.previous_config = None
        self.last_sampled_config = None

    def reset_graph(self):
        """
        Resets the planner graph to its default state.
        """
        LOGGER.info("Resetting graph!")
        self.graph.clear()
        self.graph.add_node(self.current_config)
        self.graph.add_node(self.goal_config)

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
            self.plan()

    def execute(self, delta_time, is_training):
        """
        Perform a single iteration of execution.
        """
        self.time_since_last_execute += delta_time
        # If we're not training and execution time hasn't been reached, return for now.
        if not is_training and self.time_since_last_execute < (1 / EXECUTION_FPS_LIMIT):
            return
        self.time_since_last_execute = 0
        # Check if we're at goal
        if self.current_config == self.goal_config:
            self.is_complete = True
            LOGGER.info("Got to goal!")
            self.world.save_placements_bmp(self.output_path)
            return
        # Sample the current config in the world for validity (i.e. check footpads adhere).
        if self.world.is_valid_config(self.current_config):
            self.previous_config = self.current_config.copy()
            self.current_config = next(self.execution_path, None)
        else:
            LOGGER.info("Detected invalid region! Back-tracking now.")
            self.is_executing = False
            if self.previous_config is not None:
                self.current_config = self.previous_config
                self.graph.add_node(self.current_config)
            self.world.save_placements_bmp(self.output_path)
            # TODO(mitch): IDEA: if we hit a problem, move back to the last default config.
            # would require saving every 4 configs.

    def sample(self):
        """
        Samples new config and attempts to add it to the graph.
        """
        # TODO(mitch): sample from precomputed model during playback, sample from config space
        # during training.
        LOGGER.info("Sampling!")
        sample = self.robot.get_random_config(self.world)
        if not sample.is_valid(self.world):
            return
        self.last_sampled_config = sample
        new_config_edges = []
        for config in self.graph.nodes:
            interpolations = config.interpolate(sample, self.world)
            if interpolations is None:
                continue
            new_config_edges.append((sample, config))
            interpolations = sample.interpolate(config, self.world)
            if interpolations is None:
                continue
            new_config_edges.append((config, sample))
        self.graph.add_edges_from(new_config_edges)

    def plan(self):
        """
        Perform a single iteration of planning.
        """
        if len(self.graph.nodes) >= self.GRAPH_SIZE_LIMIT:
            self.reset_graph()
        self.sample()
        # Check if there's a path through configurations starting from current to goal
        if not nx.has_path(self.graph, self.current_config, self.goal_config):
            return
        execution_path = self.get_execution_path()
        if execution_path is None:
            return
        LOGGER.info("Found a path to the goal! Executing now.")
        self.execution_path = iter(execution_path)
        self.is_executing = True

    def get_execution_path(self):
        """
        Return the complete execution path fully interpolated.
        """
        node_path = nx.shortest_path(self.graph, self.current_config, self.goal_config)
        execution_path = []
        for i in range(1, len(node_path)):
            previous, current = node_path[i - 1], node_path[i]
            interpolation = previous.interpolate(current, self.world)
            if interpolation is None:
                return None
            execution_path.extend(interpolation)
        return execution_path

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
        if self.current_config not in (self.start_config, self.goal_config):
            self.current_config.colour = EXECUTING_COLOUR
            self.current_config.draw(batch)
        pyglet.gl.glLineWidth(ROBOT_LINE_WIDTH)
        batch.draw()

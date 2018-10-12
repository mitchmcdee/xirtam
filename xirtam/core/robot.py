"""
Module containing information pertaining to the simulated robot.
"""
import logging
import math
import pyglet
import csv
import numpy as np
from operator import gt, lt
from circle import Circle
from vector2d import Vector2D
from point2d import Point2D
from random import uniform
from utils import (
    clamp,
    get_circle_call,
    get_coerced_reader_row_helper,
    get_intersection_between_two_circles,
)
from settings import NUM_FOOT_POINTS, PLANNING_COLOUR, BODY_ALT_MODIFIER
from world import World

LOGGER = logging.getLogger(__name__)
EPSILON_Z = (1e-2,)


class Robot:
    """
    A simulated robot for the virtual environment.
    """

    NUM_LEGS = 4

    def __init__(self, robot_filepath: str) -> None:
        with open(robot_filepath) as robot_file:
            robot_reader = csv.reader(robot_file)
            get_robot_row = get_coerced_reader_row_helper(robot_reader, robot_filepath)
            self.body_length = get_robot_row([float], "body length")
            self.femur_length = get_robot_row([float], "femur length")
            self.tibia_length = get_robot_row([float], "tibia length")
            self.foot_radius = get_robot_row([float], "foot radius")
            self.min_permeability = get_robot_row([float], "minimum magnetic permeability")

    def __hash__(self):
        """
        Returns the unique hash for the config.
        """
        return hash(
            (
                self.body_length,
                self.femur_length,
                self.tibia_length,
                self.foot_radius,
                self.min_permeability,
            )
        )

    def can_hold(self, material_permeability):
        """
        Returns True if the robot can support itself on the given material, else False.
        Note: this is extremely simplified due to the complicated equations required
        to calculate magnetic pull force. Since this is not the focus of the simulation,
        this simplification is acceptable.
        """
        return material_permeability >= self.min_permeability

    def get_random_config(self, world: World):
        """
        Returns the robot in a random position and of random pose within the given world.
        Note: This function provides no guarantees on the validity of the config.
        """
        left, top, right, bottom = world.bounding_box.bounds
        # Place in a random position in workspace with random heading
        position = Point2D(uniform(left, right), uniform(bottom, top))
        heading = uniform(0, 2 * math.pi)
        # Place feet in world relative to the body. Adopt a natural, equidistant foot stance.
        foot_vertices = []
        for foot_index in range(Robot.NUM_LEGS):
            # Place feet equidistantly around an octagon (skipping every second placement)
            # Note: Since we're using radians, this is relative to the unit circle, thus the
            # negative sign to ensure we're incrementing clockwise.
            angle = -(foot_index * 2 + 1) * (math.pi / 4)
            # TODO(mitch): add randomness to foot offset? Or maybe just implement backpedal idea?
            # offset = Point2D.from_angle(angle) * uniform(self.femur_length / 2, 3 * self.femur_length / 2)
            offset = Point2D.from_angle(angle + heading) * self.femur_length
            foot_vertices.append(position + offset)
        return RobotConfig(self, position, heading, foot_vertices, PLANNING_COLOUR)


class RobotConfig:
    """
    A particular config for a robot representing an explicit position and pose in space.
    """

    def __init__(self, robot, position, heading, foot_vertices, colour) -> None:
        if isinstance(position, tuple):
            position = Point2D(*position)
        # Valid until proven otherwise.
        self.valid = True
        self.robot = robot
        self.colour = colour
        # Alternate colour to aide diffentiation between sides of the robot.
        self.alt_colour = tuple([int(clamp(BODY_ALT_MODIFIER * c, 0, 255)) for c in self.colour])
        self.heading = heading
        self.position = position
        self.foot_vertices = foot_vertices
        self.footprints = self.get_footprints()
        self.hip_vertices = self.get_hip_vertices()
        self.body_vertices = self.get_body_vertices()
        self.leg_vertices = self.get_leg_vertices()

    def __eq__(self, other: "RobotConfig"):
        """
        Returns True if the two configs are the same robot, at the same position, facing the
        same way with the same foot placements.
        """
        return (
            self.robot == other.robot
            and self.heading == other.heading
            and self.position == other.position
            and all(self.footprints[i] == other.footprints[i] for i in range(Robot.NUM_LEGS))
        )

    def __hash__(self):
        """
        Returns the unique hash for the config.
        """
        return hash((self.robot, self.heading, self.position, tuple(self.footprints)))

    def copy(self):
        """
        Returns a copy of the robot config.
        """
        return RobotConfig(self.robot, self.position, self.heading, self.foot_vertices, self.colour)

    def get_footprints(self):
        """
        Returns the circle footprints for the given vertices.
        """
        return [Circle(foot_vertex, self.robot.foot_radius) for foot_vertex in self.foot_vertices]

    def interpolate_feet(self, other: "RobotConfig", world: World):
        """
        Returns the interpolated feet configs moving from the current config
        to the given other.
        Note: Assumes that the interpolation has already been validated outside of unknown invalid
        regions. This is merely interpolating the feet movement from one valid config to
        another.
        """
        configs = []  # type: List["RobotConfig"]
        last_config = self
        # Move all feet one by one, closest feet to the other position first.
        foot_order = sorted(
            [(f.centre.distance_to(other.position), i) for i, f in enumerate(self.footprints)]
        )
        for _, foot_index in foot_order:
            new_foot_vertices = [footprint.centre for footprint in last_config.footprints]
            # Move one foot.
            new_foot_vertices[foot_index] = other.footprints[foot_index].centre
            config = RobotConfig(
                self.robot, self.position, self.heading, new_foot_vertices, self.colour
            )
            if not config.is_valid(world):
                return None
            configs.append(config)
            last_config = config
        # Move body
        configs.append(other)
        return configs

    def interpolate_angle(self, other: "RobotConfig", world: World):
        """
        Returns the interpolated feet configs rotating from the current config
        to the given other.
        """
        current_vector = Vector2D.from_angle(self.heading)
        other_vector = Vector2D.from_angle(other.heading)
        rotation_angle = current_vector.directed_angle_between(other_vector)
        # TODO(mitch): review this value
        rotation_amount = math.atan2(self.robot.tibia_length / 4, self.robot.femur_length)
        configs = []  # type: List["RobotConfig"]
        last_config = self
        # Perform the minimum number of moves to get to the other config without exceeding
        # rotation limits. This should be at least two; the first move and last move.
        num_moves = max(2, math.ceil(abs(rotation_angle / rotation_amount)))
        for angle in np.linspace(0, rotation_angle, num_moves):
            new_foot_vertices = [
                footprint.centre.rotated_around_point(angle, self.position)
                for footprint in self.footprints
            ]
            config = RobotConfig(
                self.robot, self.position, self.heading + angle, new_foot_vertices, self.colour
            )
            if not config.is_valid(world):
                return None
            feet_configs = last_config.interpolate_feet(config, world)
            if feet_configs is None:
                return None
            configs.extend(feet_configs)
            last_config = configs[-1]
        return configs

    def interpolate_movement(self, other: "RobotConfig", world: World):
        """
        Returns the interpolated feet configs moving from the current config
        to the given other.
        """
        movement_vector = Vector2D.from_points(self.position, other.position)
        movement_distance = movement_vector.length
        # TODO(mitch): review this value
        movement_amount = 2 * self.robot.tibia_length / 5
        configs = []  # type: List["RobotConfig"]
        last_config = self
        # Perform the minimum number of moves to get to the other config without exceeding
        # movement limits. This should be at least two: the first move and last move.
        num_moves = max(2, math.ceil(movement_distance / movement_amount))
        for distance in np.linspace(0, movement_distance, num_moves):
            translation_vector = movement_vector.normalized * distance
            foot_placements = [
                footprint.centre + translation_vector for footprint in self.footprints
            ]
            config = RobotConfig(
                self.robot,
                self.position + translation_vector,
                self.heading,
                foot_placements,
                self.colour,
            )
            if not config.is_valid(world):
                return None
            feet_configs = last_config.interpolate_feet(config, world)
            if feet_configs is None:
                return None
            configs.extend(feet_configs)
            last_config = configs[-1]
        return configs

    def interpolate(self, goal: "RobotConfig", world: World):
        """
        Returns the interpolated configs getting from the current config to the
        given goal. If interpolation is not possible, returns None.
        """
        configs = []  # type: List["RobotConfig"]
        # Add movement to goal pose.
        movement_configs = self.interpolate_movement(goal, world)
        if movement_configs is None:
            return None
        configs.extend(movement_configs)
        # Add rotation into goal pose.
        angle_configs = configs[-1].interpolate_angle(goal, world)
        if angle_configs is None:
            return None
        configs.extend(angle_configs)
        # Add transition into goal pose.
        feet_configs = configs[-1].interpolate_feet(goal, world)
        if feet_configs is None:
            return None
        configs.extend(feet_configs)
        return configs

    def is_valid(self, world: World):
        """
        Returns True if is a config in the workspace, else False.
        """
        # Check if invalidity was detected during config construction.
        if not self.valid:
            return False
        # Check feet are within world bounds.
        for footprint in self.footprints:
            if not world.bounding_box.intersects(footprint):
                return False
        # Rotate feet back to robot's "north"-heading.
        footprints = [
            f.centre.rotated_around_point(-self.heading, self.position) for f in self.footprints
        ]
        # Check that there is a foot in every quadrant (i.e. a crude "centre" of mass" check).
        # Order: top-right, bottom-right, bottom-left, top-left.
        for i, (x_op, y_op) in enumerate([(gt, gt), (gt, lt), (lt, lt), (lt, gt)]):
            if any(x_op(f.x, self.position.x) and y_op(f.y, self.position.y) for f in footprints):
                continue
            # If no foot was found in the quadrant, invalid centre of mass.
            return False
        return True

    def get_point_along_body(self, distance_from_front: float):
        """
        Get the point at the given distance from the front of the body.
        """
        distance_to_front = self.robot.body_length / 2
        delta_from_centre = distance_to_front - distance_from_front
        point_along_body = self.position + Point2D.from_angle(self.heading) * delta_from_centre
        # Note: Here we place body at half height to adopt a somewhat natural pose.
        return (*point_along_body.coords, self.robot.tibia_length / 2)

    def get_body_vertices(self):
        """
        Get the vertices for the robot body.
        """
        return (*self.get_point_along_body(0), *self.get_point_along_body(self.robot.body_length))

    def get_hip_vertices(self):
        """
        Get the vertices for the given robot leg's hip.
        """
        num_intra_hip_segments = (Robot.NUM_LEGS // 2) - 1
        distance_between_hip_joints = self.robot.body_length / num_intra_hip_segments
        hip_vertices = []
        for leg_index in range(Robot.NUM_LEGS):
            # Assign hip indices in a circular fashion, e.g.:
            # Leg 1 -> Hip 1, Leg 2 -> Hip 2, Leg 3 -> Hip 2, Leg 4 -> Hip 1
            hip_index = (
                leg_index if leg_index < (Robot.NUM_LEGS // 2) else Robot.NUM_LEGS - leg_index - 1
            )
            hip_distance_from_front = hip_index * distance_between_hip_joints
            hip_vertex = self.get_point_along_body(hip_distance_from_front)
            hip_vertices.append(hip_vertex)
        return hip_vertices

    def get_leg_vertices(self):
        """
        Get the vertices for the given robot leg.
        """
        leg_vertices = []
        for leg_index in range(Robot.NUM_LEGS):
            # Add dummy z value to place foot in 3D above floor.
            foot_vertex = self.footprints[leg_index].centre.coords + EPSILON_Z
            hip_vertex = self.hip_vertices[leg_index]
            intersections = get_intersection_between_two_circles(
                foot_vertex, self.robot.femur_length, hip_vertex, self.robot.tibia_length
            )
            # If no intersections could be found, invalid config.
            if intersections is None:
                self.valid = False
                return None
            # Since we're working only in 2.5D, we will only ever need the highest y-coord position,
            # which is the first intersection.
            knee_vertex, _ = intersections
            leg_vertices.append((*foot_vertex, *knee_vertex, *knee_vertex, *hip_vertex))
        return leg_vertices

    def get_body_call(self):
        """
        Get the draw call for the robot body.
        """
        return (
            2,
            pyglet.gl.GL_LINES,
            None,
            ("v3f", self.body_vertices),
            ("c3B", self.alt_colour + self.colour),
        )

    def get_leg_call(self, leg_index: int):
        """
        Get the draw call for the robot leg.
        """
        # Highlight first two legs of the robot.
        colour = self.alt_colour if leg_index in (0, Robot.NUM_LEGS - 1) else self.colour
        return (
            4,
            pyglet.gl.GL_LINES,
            None,
            ("v3f", self.leg_vertices[leg_index]),
            ("c3B", colour * 4),
        )

    def get_foot_call(self, leg_index: int):
        """
        Get the draw call for the robot foot.
        """
        # Add dummy z value to place foot in 3D above floor.
        foot_vertex = self.footprints[leg_index].centre.coords + EPSILON_Z
        # Highlight first two feet of the robot.
        colour = self.alt_colour if leg_index in (0, Robot.NUM_LEGS - 1) else self.colour
        return get_circle_call(
            *foot_vertex, self.robot.foot_radius, NUM_FOOT_POINTS, colour
        )  # type: ignore

    def draw(self, batch):
        """
        Adds the robot pose to the provided draw batch.
        """
        # Body
        batch.add(*self.get_body_call())
        # Legs and feet
        for leg_index in range(Robot.NUM_LEGS):
            batch.add(*self.get_leg_call(leg_index))
            batch.add_indexed(*self.get_foot_call(leg_index))

"""
Module containing world information for the simulation environment.
"""
import os
import csv
import logging
import pyglet
import hashlib
from PIL import Image, ImageDraw
from enum import Enum
from itertools import cycle
from rectangle import Rectangle
from point2d import Point2D
from utils import get_coerced_reader_row_helper, get_circle_call, translate, get_translated_bounds
from settings import (
    INVALID_COLOUR,
    VALID_COLOUR,
    NUM_FOOT_POINTS,
    MIN_PERMEABILITY_COLOUR,
    MAX_PERMEABILITY_COLOUR,
    OUTPUT_BMP_DIMENSIONS,
    OUTPUT_DEFAULT_COLOUR,
    OUTPUT_VALID_COLOUR,
    OUTPUT_INVALID_COLOUR,
)

LOGGER = logging.getLogger(__name__)
EPSILON_Z = 1e-2


class WorldRegion(Rectangle):
    """
    A region within the world.
    """

    def __init__(self, permeability, x, y, width, height):
        super().__init__(x, y, width, height)
        self.permeability = permeability


class WorldVisibilityState(Enum):
    """
    Visibility state of the world.
    """

    ALL = 0
    OFF = 1
    PLACEMENTS = 2
    REGIONS = 3


class World:
    """
    An interactable virtual world.
    """

    def __init__(self, world_filepath):
        # Parse world file
        with open(world_filepath) as world_file:
            world_reader = csv.reader(world_file)
            get_world_row = get_coerced_reader_row_helper(world_reader, world_filepath)
            self.bounding_box = Rectangle(*get_world_row([float] * 4, "world bounding box"))
            num_region = get_world_row([int], "# of regions")
            self.regions = []
            for _ in range(num_region):
                permeability = get_world_row([float], "region permeability")
                region_bounding_box = get_world_row([float] * 4, "region bounding box")
                self.regions.append(WorldRegion(permeability, *region_bounding_box))
        self.visibility_iterator = cycle(WorldVisibilityState)
        self.visibility_state = next(self.visibility_iterator)
        self.initialise()

    def __hash__(self):
        """
        Returns the unique hash for the world.
        """
        return hash((self.bounding_box, tuple(self.regions)))

    def initialise(self):
        """
        Initialise world.
        """
        self.valid_placements = []
        self.invalid_placements = []
        self.region_batch = pyglet.graphics.Batch()
        self.placements_batch = pyglet.graphics.Batch()
        # Add all region to batch draw
        permeablities = [region.permeability for region in self.regions]
        min_permeability = min(permeablities)
        max_permeability = max(permeablities)
        for region in self.regions:
            colour = int(
                translate(
                    region.permeability,
                    min_permeability,
                    max_permeability,
                    MIN_PERMEABILITY_COLOUR,
                    MAX_PERMEABILITY_COLOUR,
                )
            )
            self.region_batch.add(
                4, pyglet.gl.GL_QUADS, None, ("v2f", region.vertices), ("c3B", (colour,) * 3 * 4)
            )

    def handle_reset(self):
        """
        Handle the user attempting to reset the simulation.
        """
        self.initialise()

    def handle_toggle_world(self):
        """
        Handle the user attempting to toggle the world view of the simulation.
        """
        self.visibility_state = next(self.visibility_iterator)

    def is_valid_config(self, config):
        """
        Sample the world with the given robot configuration. If an invalid region was sampled,
        add it to the collection of invalid regions. Returns True if the sample was valid, else
        False.
        """
        is_valid_sample = True
        for footprint in config.footprints:
            is_valid_placement = True
            for region in self.regions:
                if config.robot.can_hold(region.permeability):
                    continue
                # We place this check second as it is vastly more computational expensive, albeit
                # more logical to check first.
                if not footprint.intersects(region):
                    continue
                is_valid_placement = False
                is_valid_sample = False
                break
            # Add to appropriate foot placements list.
            if is_valid_placement:
                self.valid_placements.append(footprint)
            else:
                self.invalid_placements.append(footprint)
            self.placements_batch.add_indexed(
                *get_circle_call(
                    *footprint.centre.coords,
                    EPSILON_Z,
                    config.robot.foot_radius,
                    NUM_FOOT_POINTS,
                    VALID_COLOUR if is_valid_placement else INVALID_COLOUR,
                )
            )
        return is_valid_sample

    def save_placements_bmp(self, robot):
        """
        Saves the foot placements as a bitmap file.
        """
        image = Image.new("L", OUTPUT_BMP_DIMENSIONS)
        draw = ImageDraw.Draw(image)
        pixels = image.load()
        output_width, output_height = image.size
        output_bounds = (0, output_height, output_width, 0)
        # Set default colour
        for i in range(output_width):
            for j in range(output_height):
                pixels[i, j] = OUTPUT_DEFAULT_COLOUR
        # Add valid placements
        for placement in self.valid_placements:
            translated_bounds = get_translated_bounds(
                placement.bounds, self.bounding_box.bounds, output_bounds
            )
            left, top, right, bottom = list(map(int, translated_bounds))
            draw.ellipse((bottom, left, top, right), fill=OUTPUT_VALID_COLOUR)
        # Add invalid placements
        for placement in self.invalid_placements:
            translated_bounds = get_translated_bounds(
                placement.bounds, self.bounding_box.bounds, output_bounds
            )
            left, top, right, bottom = list(map(int, translated_bounds))
            draw.ellipse((bottom, left, top, right), fill=OUTPUT_INVALID_COLOUR)
        # Make world-robot directory if it doesn't already exist
        world_robot_directory = f"generated_output/world-{self.__hash__()}/robot-{robot.__hash__()}"
        if not os.path.exists(world_robot_directory):
            os.makedirs(world_robot_directory)
        # Save unique image if it doesn't already exist
        image_hash = hashlib.sha512(image.tobytes()).hexdigest()
        placements_filepath = world_robot_directory + f"/{image_hash}.bmp"
        if not os.path.exists(placements_filepath):
            image.save(world_robot_directory + f"/{image_hash}.bmp")
            LOGGER.info("Saved placements!")

    def save_regions_bmp(self, robot):
        """
        Saves the regions as a bitmap file.
        """
        image = Image.new("L", OUTPUT_BMP_DIMENSIONS)
        draw = ImageDraw.Draw(image)
        pixels = image.load()
        output_width, output_height = image.size
        output_bounds = (0, output_height, output_width, 0)
        # Set default colour
        for i in range(output_width):
            for j in range(output_height):
                pixels[i, j] = OUTPUT_DEFAULT_COLOUR
        # Add regions
        for region in self.regions:
            translated_bounds = get_translated_bounds(
                region.bounds, self.bounding_box.bounds, output_bounds
            )
            left, top, right, bottom = list(map(int, translated_bounds))
            if robot.can_hold(region.permeability):
                colour = OUTPUT_VALID_COLOUR
            else:
                colour = OUTPUT_INVALID_COLOUR
            draw.rectangle((bottom, left, top, right), fill=colour)
        # Make world-robot directory if it doesn't already exist
        world_robot_directory = f"generated_output/world-{self.__hash__()}/robot-{robot.__hash__()}"
        if not os.path.exists(world_robot_directory):
            os.makedirs(world_robot_directory)
        image.save(world_robot_directory + "/regions.bmp")
        LOGGER.info("Saved regions!")

    def draw(self):
        """
        Draw the world region.
        """
        if self.visibility_state in (WorldVisibilityState.ALL, WorldVisibilityState.REGIONS):
            self.region_batch.draw()
        if self.visibility_state in (WorldVisibilityState.ALL, WorldVisibilityState.PLACEMENTS):
            self.placements_batch.draw()

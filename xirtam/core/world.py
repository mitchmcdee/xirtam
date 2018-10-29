"""
Module containing world information for the simulation environment.
"""
import os
import csv
from logging import getLogger
from pyglet.graphics import Batch
from pyglet.gl import GL_QUADS
from hashlib import sha512
from functools import partial
from typing import List  # noqa: F401
from PIL import Image, ImageDraw
from xirtam.utils.geometry.circle import Circle  # noqa: F401
from xirtam.utils.geometry.rectangle import Rectangle
from xirtam.utils.utils import (
    get_coerced_reader_row_helper,
    get_circle_call,
    translate,
    get_translated_bounds,
)
from xirtam.core.settings import (
    INVALID_PLACEMENT_COLOUR,
    VALID_PLACEMENT_COLOUR,
    NUM_FOOT_POINTS,
    MIN_PERMEABILITY_COLOUR,
    MAX_PERMEABILITY_COLOUR,
    OUTPUT_BMP_DIMENSIONS,
    OUTPUT_DEFAULT_COLOUR,
    OUTPUT_VALID_COLOUR,
    OUTPUT_INVALID_COLOUR,
)

LOGGER = getLogger(__name__)
EPSILON_Z = 1e-2


class WorldRegion(Rectangle):
    """
    A region within the world.
    """

    def __init__(self, permeability, x, y, width, height):
        super().__init__(x, y, width, height)
        self.permeability = permeability

    def __eq__(self, other) -> bool:
        """
        Returns True if the regions are the same rectangle with the same permeability, else False.
        """
        return super().__eq__(other) and self.permeability == other.permeability

    def __hash__(self):
        """
        Returns the unique hash for the region.
        """
        return hash((super().__hash__(), self.permeability))


class World(Rectangle):
    """
    An interactable virtual world.
    """

    # Collection of region tiles in the world.
    regions = []  # type: List[WorldRegion]
    # Collection of valid foot placements in the world.
    valid_placements = []  # type: List[Circle]
    # Collection of invalid foot placements in the world.
    invalid_placements = []  # type: List[Circle]
    # Region draw batch.
    region_batch = Batch()
    # Region draw batch.
    placements_batch = Batch()

    def __init__(self, world_path):
        # Parse world file
        with open(world_path) as world_file:
            world_reader = csv.reader(world_file)
            get_world_row = get_coerced_reader_row_helper(world_reader, world_path)
            super().__init__(*get_world_row([float] * 4, "world bounding box"))
            num_region = get_world_row([int], "# of regions")
            for _ in range(num_region):
                permeability = get_world_row([float], "region permeability")
                region_bounding_box = get_world_row([float] * 4, "region bounding box")
                self.regions.append(WorldRegion(permeability, *region_bounding_box))
        self.initialise()

    def __hash__(self):
        """
        Returns the unique hash for the world.
        """
        return hash((super().__hash__(), tuple(self.regions)))

    def initialise(self):
        """
        Initialise world.
        """
        self.valid_placements = []
        self.invalid_placements = []
        self.region_batch = Batch()
        self.placements_batch = Batch()
        # Add all region to batch draw
        permeablities = [region.permeability for region in self.regions]
        translate_permeability = partial(
            translate,
            left_min=min(permeablities),
            left_max=max(permeablities),
            right_min=MIN_PERMEABILITY_COLOUR,
            right_max=MAX_PERMEABILITY_COLOUR,
        )
        for region in self.regions:
            colours = (int(translate_permeability(region.permeability)),) * 3 * 4
            self.region_batch.add(4, GL_QUADS, None, ("v2f", region.vertices), ("c3B", colours))

    def handle_reset(self):
        """
        Handle the user attempting to reset the simulation.
        """
        self.initialise()

    def is_valid_config(self, config, update=True):
        """
        Sample the world with the given robot configuration. If an invalid region was sampled,
        add it to the collection of invalid regions if `update` is set to True. Returns True
        if the sample was valid, else False.
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
            if update:
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
                        VALID_PLACEMENT_COLOUR if is_valid_placement else INVALID_PLACEMENT_COLOUR,
                    )
                )
        return is_valid_sample

    def intersects_invalid(self, config):
        """
        Returns True if the config intersects with an already discovered invalid region, else False.
        """
        for footprint in config.footprints:
            for invalid_placement in self.invalid_placements:
                if not footprint.intersects(invalid_placement):
                    continue
                # Check if footprint is exactly on a previous valid placement. If not, its invalid.
                for valid_placement in self.valid_placements:
                    if footprint == valid_placement:
                        break
                else:
                    return True
        return False

    def get_placements_bmp(self, robot):
        """
        Gets the robot valid and invalid foot placements as a bitmap image.
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
            translated_bounds = get_translated_bounds(placement.bounds, self.bounds, output_bounds)
            left, top, right, bottom = list(map(int, translated_bounds))
            draw.ellipse((left, bottom, right, top), fill=OUTPUT_VALID_COLOUR)
        # Add invalid placements
        for placement in self.invalid_placements:
            translated_bounds = get_translated_bounds(placement.bounds, self.bounds, output_bounds)
            left, top, right, bottom = list(map(int, translated_bounds))
            draw.ellipse((left, bottom, right, top), fill=OUTPUT_INVALID_COLOUR)
        return image

    def save_placements_bmp(self, robot, output_directory):
        """
        Saves the robot valid and invalid foot placements as a bitmap file.
        """
        image = self.get_placements_bmp(robot)
        # Save unique image if it doesn't already exist
        image_hash = sha512(image.tobytes()).hexdigest()
        placements_path = os.path.join(output_directory, f"{image_hash}.bmp")
        if os.path.exists(placements_path):
            return
        # Make ouput directory if it doesn't already exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        image.save(placements_path)
        LOGGER.info("Saved placements!")
        # Save regions for the given placement if it doesn't exist.
        self.save_regions_bmp(robot, output_directory)

    def save_regions_bmp(self, robot, output_directory):
        """
        Saves the regions as a bitmap file.
        """
        # Make ouput directory if it doesn't already exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        regions_path = os.path.join(output_directory, "regions.bmp")
        if os.path.exists(regions_path):
            return
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
            translated_bounds = get_translated_bounds(region.bounds, self.bounds, output_bounds)
            left, top, right, bottom = list(map(int, translated_bounds))
            if robot.can_hold(region.permeability):
                colour = OUTPUT_VALID_COLOUR
            else:
                colour = OUTPUT_INVALID_COLOUR
            draw.rectangle((left, bottom, right, top), fill=colour)
        image.save(regions_path)
        LOGGER.info("Saved regions!")

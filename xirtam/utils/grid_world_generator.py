"""
Module containing high level class for generating grid world files.
"""
import os
import sys
import math
import argparse
import numpy as np
import networkx as nx
from xirtam.utils.utils import translate
from xirtam.core.robot import Robot
from xirtam.core.world import World
from random import randint, randrange, choice

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parse_args(args):
    """
    Parses world generator CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--side_cells",
        type=int,
        help="Number of cells in each row/column of grid",
        default=randint(5, 25),
    )
    parser.add_argument(
        "-p",
        "--invalid_percentage",
        type=int,
        help="Percentage of cells in world which are invalid",
        default=0.3,
    )
    parser.add_argument(
        "-w",
        "--world_output_path",
        type=str,
        help="Path to world output file",
        default="./test.world",
    )
    parser.add_argument(
        "-m",
        "--motion_output_path",
        type=str,
        help="Path to motion output file",
        default="./test.motion",
    )
    parser.add_argument(
        "-r", "--robot_path", type=str, help="Path to robot file", default="./magneto.robot"
    )
    parser.add_argument(
        "-g", "--num_genesis_cells", type=int, help="Number of genesis cells in grid", default=3
    )
    parser.add_argument(
        "-i", "--invalid_perm", type=float, help="Invalid permeability amount", default=1.0e-10
    )
    parser.add_argument(
        "-v", "--valid_perm", type=float, help="Valid permeability amount", default=1.0
    )
    parser.add_argument(
        "-l", "--side_length", type=float, help="Side length of each cell", default=10.0
    )
    return parser.parse_args(args)


def process_generation_args(test_id, *args, **kwargs):
    """
    Processes simulation args to determine whether test case generation is required.
    Performs generation if necessary and returns the modified args.
    """
    is_generating = kwargs.pop("generate")
    world_path = kwargs.get("world_path")
    motion_path = kwargs.get("motion_path")
    robot_path = kwargs.get("robot_path")
    if world_path is None or motion_path is None:
        output_path = kwargs.get("output_path")
        temp_directory = os.path.join(output_path, "tmp/")
        # Make temp directory if it doesn't already exist
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        world_path = os.path.join(temp_directory, f"training-{test_id}.world")
        motion_path = os.path.join(temp_directory, f"training-{test_id}.motion")
        kwargs["world_path"] = world_path
        kwargs["motion_path"] = motion_path
    if is_generating:
        generator_args = ["-w", world_path, "-m", motion_path, "-r", robot_path]
        grid_generator(generator_args)
    return args, kwargs


def generate_world(side_cells, invalid_percentage, num_genesis_cells):
    """
    Generates world cells to the given paramaters.
    """

    def is_bad_cell(cell):
        cell_x, cell_y = cell
        return cell_x in (-1, side_cells) or cell_y in (-1, side_cells) or cell in invalid_cells

    invalid_cells = []
    # Add genesis cells
    for _ in range(num_genesis_cells):
        invalid_cells.append((randrange(0, side_cells), randrange(0, side_cells)))
    max_invalid_cells = int(invalid_percentage * side_cells * side_cells)
    # Place invalid cells while under limit.
    while len(invalid_cells) < max_invalid_cells:
        seed_x, seed_y = choice(invalid_cells)
        delta_x, delta_y = choice(DIRECTIONS)
        invalid_cell = (seed_x + delta_x, seed_y + delta_y)
        invalid_x, invalid_y = invalid_cell
        if is_bad_cell(invalid_cell):
            continue
        invalid_cells.append(invalid_cell)
        # Fill any resulting voids.
        for x, y in DIRECTIONS:
            neighbour_cell = (invalid_x + x, invalid_y + y)
            neighbour_x, neighbour_y = neighbour_cell
            if is_bad_cell(neighbour_cell):
                continue
            # Check neighbour cell is surrounded indeed a void.
            for x, y in DIRECTIONS:
                outer_cell = (neighbour_x + x, neighbour_y + y)
                outer_x, outer_y = outer_cell
                if is_bad_cell(outer_cell):
                    continue
                break
            # no break, neighbour cell is a void cell.
            else:
                invalid_cells.append(neighbour_cell)
    # Draw up grid and return. Reverse row order to make bottom left the origin.
    return [
        [1 if (i, j) in invalid_cells else 0 for i in range(side_cells)]
        for j in reversed(range(side_cells))
    ]


def save_world(world_cells, output_path, side_length, valid_perm, invalid_perm):
    """
    Writes the given world to its output file.
    """
    with open(output_path, "w") as output_file:
        x, y = (0, 0)
        side_cells = len(world_cells)
        cell_size = side_length / side_cells
        # World left, bottom, width, height
        output_file.write(f"{x}, {y}, {side_length}, {side_length}\n")
        # Number of cells (i.e. regions)
        output_file.write(f"{side_cells ** 2}\n")
        for j, bottom in enumerate(np.linspace(y, y + side_length, side_cells, endpoint=False)):
            for i, left in enumerate(np.linspace(x, x + side_length, side_cells, endpoint=False)):
                # Region permeability
                permeability = invalid_perm if world_cells[j][i] else valid_perm
                output_file.write(f"{permeability}\n")
                # Region left, bottom, width, height
                output_file.write(f"{left}, {bottom}, {cell_size}, {cell_size}\n")


def generate_motion(robot, world, world_cells):
    """
    Generates a random motion plan for the given robot in the given world.
    """
    side_cells = len(world_cells)
    # Build world graph for cell connectivity.
    graph = nx.Graph()
    for x in range(side_cells):
        for y in range(side_cells):
            cell = (x, y)
            if world_cells[y][x]:
                continue
            for delta_x, delta_y in DIRECTIONS:
                neighbour = (x + delta_x, y + delta_y)
                neighbour_x, neighbour_y = neighbour
                if neighbour_x in (-1, side_cells) or neighbour_y in (-1, side_cells):
                    continue
                if world_cells[neighbour_y][neighbour_x]:
                    continue
                graph.add_edge(cell, neighbour)
    # Get valid start config.
    start_config = goal_config = None
    while True:
        start_config = robot.get_random_config(world)
        # Check valid cell.
        if start_config.is_valid(world) and world.is_valid_config(start_config, update=False):
            break
    # Get valid goal config and ensure it can be reached by the start.
    while True:
        goal_config = robot.get_random_config(world)
        # Check valid cell.
        if not goal_config.is_valid(world) or not world.is_valid_config(goal_config, update=False):
            continue
        # Check we can interpolate to it (primitive check to see if accessible via rotation).
        if start_config.interpolate(goal_config, world) is None:
            continue
        world_left, world_top, world_right, world_bottom = world.bounds
        start_x = int(translate(start_config.x, world_left, world_right, 0, side_cells))
        start_y = int(translate(start_config.y, world_bottom, world_top, 0, side_cells))
        start_cell = (start_x, start_y)
        goal_x = int(translate(goal_config.x, world_left, world_right, 0, side_cells))
        goal_y = int(translate(goal_config.y, world_bottom, world_top, 0, side_cells))
        goal_cell = (goal_x, goal_y)
        if nx.has_path(graph, start_cell, goal_cell):
            if len(nx.shortest_path(graph, start_cell, goal_cell)) > 2:
                break
    return start_config, goal_config


def save_motion(motion_plan, output_path):
    """
    Writes the given motion plan to its output file.
    """
    with open(output_path, "w") as output_file:
        for config in motion_plan:
            # Config position
            position_x, position_y = config.x, config.y
            output_file.write(f"{position_x}, {position_y}\n")
            # Config heading (in degrees)
            output_file.write(f"{math.degrees(config.heading)}\n")
            # Config foot vertices
            for foot_vertex in config.foot_vertices:
                vertex_x, vertex_y = foot_vertex.coords
                output_file.write(f"{vertex_x}, {vertex_y}\n")


def grid_generator(generator_args):
    """
    Generates the grid world.
    """
    args = parse_args(generator_args)
    robot = Robot(args.robot_path)
    world_cells = generate_world(args.side_cells, args.invalid_percentage, args.num_genesis_cells)
    save_world(
        world_cells, args.world_output_path, args.side_length, args.valid_perm, args.invalid_perm
    )
    world = World(args.world_output_path)
    motion_plan = generate_motion(robot, world, world_cells)
    save_motion(motion_plan, args.motion_output_path)


def main():
    grid_generator(sys.argv[1:])


if __name__ == "__main__":
    main()

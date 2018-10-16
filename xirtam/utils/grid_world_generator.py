"""
Module containing high level class for generating grid world files.
"""
import os
import sys
import math
import argparse
import numpy as np
import networkx as nx
from xirtam.core.robot import Robot
from xirtam.core.world import World
from random import randrange, choice

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parse_args(args):
    """
    Parses world generator CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--robot_filepath", type=str, help="Path to robot file", default="./test.robot"
    )
    parser.add_argument(
        "-w",
        "--world_output_filepath",
        type=str,
        help="Path to world output file",
        default="./test.world",
    )
    parser.add_argument(
        "-m",
        "--motion_output_filepath",
        type=str,
        help="Path to motion output file",
        default="./test.motion",
    )
    parser.add_argument(
        "-mi",
        "--max_invalid_cells",
        type=int,
        help="Maximum number of invalid cells in grid",
        default=30,
    )
    parser.add_argument(
        "-g",
        "--num_genesis_cells",
        type=int,
        help="Number of genesis cells in grid",
        default=3,
    )
    parser.add_argument(
        "-i", "--invalid_perm", type=float, help="Invalid permeability amount", default=1.0e-10
    )
    parser.add_argument(
        "-v", "--valid_perm", type=float, help="Valid permeability amount", default=1.0
    )
    parser.add_argument("-c", "--cell_size", type=float, help="Grid cell size", default=1.0)
    parser.add_argument("-nr", "--num_rows", type=int, help="Number of rows in grid", default=10)
    parser.add_argument("-nc", "--num_cols", type=int, help="Number of columns in grid", default=10)
    return parser.parse_args(args)


def process_generation_args(test_id, *args, **kwargs):
    """
    Processes simulation args to determine whether test case generation is required.
    Performs generation if necessary and returns the modified args.
    """
    is_generating = kwargs.pop("generate_case")
    if is_generating:
        robot_filepath = kwargs.get("robot_filepath")
        world_filepath = kwargs.get("world_filepath")
        motion_filepath = kwargs.get("motion_filepath")
        output_filepath = kwargs.get("output_filepath")
        temp_directory = os.path.join(output_filepath, "tmp/")
        # Make temp directory if it doesn't already exist
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        world_filepath = os.path.join(temp_directory, f"training-{test_id}.world")
        motion_filepath = os.path.join(temp_directory, f"training-{test_id}.motion")
        kwargs["world_filepath"] = world_filepath
        kwargs["motion_filepath"] = motion_filepath
        generator_args = ["-w", world_filepath, "-m", motion_filepath, "-r", robot_filepath]
        grid_generator(generator_args)
    return args, kwargs


def generate_world(num_rows, num_cols, max_invalid_cells, num_genesis_cells):
    """
    Generates world cells to the given paramaters.
    """
    invalid_cells = []
    # Add genesis cells
    for _ in range(num_genesis_cells):
        invalid_cells.append((randrange(0, num_cols), randrange(0, num_rows)))
    is_bad_cell = lambda c: c[0] in (-1, num_cols) or c[1] in (-1, num_rows) or c in invalid_cells
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
    # Draw up grid and return.
    return [[1 if (i, j) in invalid_cells else 0 for i in range(num_cols)] for j in range(num_rows)]


def save_world(world_cells, output_filepath, cell_size, valid_perm, invalid_perm):
    """
    Writes the given world to its output file.
    """
    with open(output_filepath, "w") as output_file:
        x, y = (0, 0)
        num_rows = len(world_cells)
        num_cols = len(world_cells[0])
        width = num_cols * cell_size
        height = num_rows * cell_size
        # World left, bottom, width, height
        output_file.write(f"{x}, {y}, {width}, {height}\n")
        # Number of cells (i.e. regions)
        output_file.write(f"{num_rows * num_cols}\n")
        for j, bottom in enumerate(np.linspace(y, y + height, num_rows, endpoint=False)):
            for i, left in enumerate(np.linspace(x, x + width, num_cols, endpoint=False)):
                # Region permeability
                permeability = invalid_perm if world_cells[j][i] else valid_perm
                output_file.write(f"{permeability}\n")
                # Region left, bottom, width, height
                output_file.write(f"{left}, {bottom}, {cell_size}, {cell_size}\n")


def generate_motion(robot, world, world_cells):
    """
    Generates a random motion plan for the given robot in the given world.
    """
    num_rows = len(world_cells)
    num_cols = len(world_cells[0])
    # Build world graph for cell connectivity.
    graph = nx.Graph()
    for x in range(num_cols):
        for y in range(num_rows):
            cell = (x, y)
            if world_cells[y][x]:
                continue
            graph.add_node(cell)
            for delta_x, delta_y in DIRECTIONS:
                neighbour = (x + delta_x, y + delta_y)
                neighbour_x, neighbour_y = neighbour
                if neighbour_x in (-1, num_cols) or neighbour_y in (-1, num_rows):
                    continue
                if world_cells[neighbour_y][neighbour_x]:
                    continue
                graph.add_node(neighbour)
                graph.add_edge(cell, neighbour)
    # Get valid start config.
    start_config = goal_config = None
    while True:
        start_config = robot.get_random_config(world)
        if start_config.is_valid(world) and world.is_valid_config(start_config):
            break
    # Get valid goal config and ensure it can be reached by the start.
    while True:
        goal_config = robot.get_random_config(world)
        if not goal_config.is_valid(world) or not world.is_valid_config(goal_config):
            continue
        width = world.bounding_box.width
        height = world.bounding_box.height
        start_cell_x = int((start_config.position.x / width) * num_cols)
        start_cell_y = int((start_config.position.y / height) * num_rows)
        start_cell = start_cell_x, start_cell_y
        goal_cell_x = int((goal_config.position.x / width) * num_cols)
        goal_cell_y = int((goal_config.position.y / height) * num_rows)
        goal_cell = goal_cell_x, goal_cell_y
        if nx.has_path(graph, start_cell, goal_cell):
            break
    return start_config, goal_config


def save_motion(motion_plan, output_filepath):
    """
    Writes the given motion plan to its output file.
    """
    with open(output_filepath, "w") as output_file:
        for config in motion_plan:
            # Config position
            position_x, position_y = config.position.coords
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
    robot = Robot(args.robot_filepath)
    world_cells = generate_world(
        args.num_rows, args.num_cols, args.max_invalid_cells, args.num_genesis_cells
    )
    save_world(
        world_cells, args.world_output_filepath, args.cell_size, args.valid_perm, args.invalid_perm
    )
    world = World(args.world_output_filepath)
    motion_plan = generate_motion(robot, world, world_cells)
    save_motion(motion_plan, args.motion_output_filepath)


def main():
    grid_generator(sys.argv[1:])


if __name__ == "__main__":
    main()

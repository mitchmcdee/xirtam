"""
Module containing high level class for generating grid world files.
"""
import sys
import math
import argparse
import numpy as np
import networkx as nx
from xirtam.core.robot import Robot
from xirtam.core.world import World
from random import randrange, choice

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parse_args():
    """
    Parses world generator CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--robot_path", type=str, help="Path to robot file", default="./test.robot"
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
    return parser.parse_args()


def generate_world(num_rows=10, num_cols=10, max_invalid_cells=25):
    """
    Generates world cells to the given paramaters.
    """
    genesis_cell = (randrange(0, num_cols), randrange(0, num_rows))
    invalid_cells = [genesis_cell]
    # Place invalid cells.
    while len(invalid_cells) < max_invalid_cells:
        seed_x, seed_y = choice(invalid_cells)
        delta_x, delta_y = choice(DIRECTIONS)
        invalid_cell = (seed_x + delta_x, seed_y + delta_y)
        invalid_x, invalid_y = invalid_cell
        if invalid_x in (-1, num_cols) or invalid_y in (-1, num_rows):
            continue
        if invalid_cell in invalid_cells:
            continue
        invalid_cells.append(invalid_cell)
        # Fill any resulting voids.
        for invalid_delta_x, invalid_delta_y in DIRECTIONS:
            neighbour_cell = (invalid_x + invalid_delta_x, invalid_y + invalid_delta_y)
            neighbour_x, neighbour_y = neighbour_cell
            if neighbour_x in (-1, num_cols) or neighbour_y in (-1, num_rows):
                continue
            if neighbour_cell in invalid_cells:
                continue
            if any(
                (
                    (neighbour_x + x, neighbour_y + y) not in invalid_cells
                    and neighbour_x + x not in (-1, num_cols)
                    and neighbour_y + y not in (-1, num_rows)
                )
                for x, y in DIRECTIONS
            ):
                continue
            invalid_cells.append(neighbour_cell)
            if len(invalid_cells) >= max_invalid_cells:
                break
    # Draw up grid and return.
    return [[1 if (i, j) in invalid_cells else 0 for i in range(num_cols)] for j in range(num_rows)]


def write_world(world_cells, output_filepath, cell_size=1.0, valid_perm=1.0, invalid_perm=1.0e-10):
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
            graph.add_node(cell)
            for delta_x, delta_y in DIRECTIONS:
                neighbour = (x + delta_x, y + delta_y)
                neighbour_x, neighbour_y = neighbour
                if neighbour_x in (-1, num_cols) or neighbour_y in (-1, num_rows):
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


def write_motion(motion_plan, output_filepath):
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


def main():
    args = parse_args()
    robot = Robot(args.robot_path)
    world_cells = generate_world()
    write_world(world_cells, args.world_output_path)
    world = World(args.world_output_path)
    motion_plan = generate_motion(robot, world, world_cells)
    write_motion(motion_plan, args.motion_output_path)


if __name__ == "__main__":
    main()

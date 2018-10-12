"""
Module containing high level class for generating grid world files.
"""
import sys
import numpy as np


def main():
    # Get grid world specifications
    num_rows = int(input("Enter number of rows: "))
    num_columns = int(input("Enter number of columns: "))
    grid_cell_size = float(input("Enter grid cell size in metres: "))
    # TODO(mitch): add set/random permeability?
    order = input("Enter region permeabilities going from left to right, top to bottom: ")
    order = list(map(float, order.strip().split()))
    if len(order) != num_rows * num_columns:
        print("Number of region permeabilities must equal number of cells in grid")
        sys.exit(1)
    output_filepath = input("Enter output filepath: ")
    # Create grid world
    with open(output_filepath, "w") as output_file:
        x, y = (0, 0)
        width = num_columns * grid_cell_size
        height = num_rows * grid_cell_size
        # Bounding box
        output_file.write(f"{x}, {y}, {width}, {height}\n")
        # Num cells (i.e. regions)
        output_file.write(f"{num_rows * num_columns}\n")
        cell_num = 0
        for i in np.linspace(x, x + width, num_columns, endpoint=False):
            for j in np.linspace(y, y + height, num_rows, endpoint=False):
                # Permeability
                # TODO(mitch): add set/random permeability?
                output_file.write(f"{order[cell_num]}\n")
                # Region bounding box
                output_file.write(f"{i}, {j}, {grid_cell_size}, {grid_cell_size}\n")
                cell_num += 1


if __name__ == "__main__":
    main()

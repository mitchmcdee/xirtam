"""
Module containing utility functions for convenient operations.
"""
import math
import pyglet
from functools import partial
from itertools import chain
from typing import List, Any


class FileFormatException(Exception):
    """
    Raised when a file has incorrect format during parsing.
    """

    def __init__(self, line_number: float, filepath: str, error: str) -> None:
        self.message = f"Line {line_number} of '{filepath}': {error}"
        super().__init__(self.message)


def get_circle_call(x: float, y: float, z: float, r: float, num_vertices, colour):
    """
    Pyglet utility function that returns a circle of radius r in three dimensional space.
    Resolution is specified by the number of vertices on the circumference of the circle.
    Note: This method only supports placing a circle on the XY plane.
    """
    colours = (colour) * (num_vertices + 2)
    delta_theta = 360.0 / num_vertices
    # Triangle index order.
    indices = list(chain.from_iterable((0, i - 1, i) for i in range(2, num_vertices + 1)))
    # Add end of fan.
    indices.extend((0, 1, num_vertices))
    # Start of fan (centre).
    vertices = [x, y, z]
    # Add vertices around the circumference.
    for i in range(num_vertices):
        angle = math.radians(i * delta_theta)
        vertices.extend((x + r * math.cos(angle), y + r * math.sin(angle), z))
    # Add end of fan.
    vertices.extend((x + r, y, z))
    return (
        num_vertices + 2,
        pyglet.gl.GL_TRIANGLES,
        None,
        indices,
        ("v3f", vertices),
        ("c3B", colours),
    )


def get_coerced_reader_row_helper(reader, filepath: str):
    """
    Helper function that returns a partially filled out coerced reader row function.
    """
    return partial(get_coerced_reader_row, reader=reader, filepath=filepath)


def get_coerced_reader_row(
    types: List[Any], description: str, reader, filepath: str, validity_fn=lambda *_: True
):
    """
    Attempts to coerce the given CSV reader row to the given list of types. If there are no
    more rows from the reader, raises FileFormatException for the missing expected row. If the
    row items could not be coerced, raises FileFormatException for the invalid values. If the
    coerced items cannot pass the given validity function, raises FileFormatException for the
    invalid values. Else, returns the coerced items.

    """
    row = next(reader, None)
    if row is None:
        raise FileFormatException(reader.line_num, filepath, "Missing " + description)
    try:
        row = tuple([types[i](element) for i, element in enumerate(row)])
    except:
        raise FileFormatException(reader.line_num, filepath, "Could not coerce " + description)
    # If there's only one element, bring it out of the list.
    items = row[0] if len(row) == 1 else row
    if len(row) != len(types) or not validity_fn(items):
        raise FileFormatException(reader.line_num, filepath, "Invalid " + description)
    return items


def get_nowait_or_default(queue, default=None):
    """
    Attempts to get the next value from the given queue. If successful, returns
    the retrieved value, else returns the default value.
    """
    try:
        return queue.get_nowait()
    except:
        return default


def clamp(value, smallest, largest):
    """
    Limits value to the range(smallest, largest).
    """
    return max(smallest, min(value, largest))


def get_translated_bounds(bounds, left_bounds, right_bounds):
    """
    Returns the bounds from within the left bounds range translated into the right bounds range.
    """
    left, top, right, bottom = bounds
    left_left, left_top, left_right, left_bottom = left_bounds
    right_left, right_top, right_right, right_bottom = right_bounds
    left = translate(left, left_left, left_right, right_left, right_right)
    top = translate(top, left_bottom, left_top, right_bottom, right_top)
    right = translate(right, left_left, left_right, right_left, right_right)
    bottom = translate(bottom, left_bottom, left_top, right_bottom, right_top)
    return (left, top, right, bottom)


def translate(value, left_min, left_max, right_min, right_max):
    """
    Returns the value from within the left range translated into the right range.
    """
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min
    # To prevent a divide by zero, just return right_max
    if left_span == 0:
        return right_max
    # Convert the left range into a 0-1 range (float)
    scaled_value = float(value - left_min) / float(left_span)
    # Convert the 0-1 range into a value in the right range.
    return right_min + (scaled_value * right_span)


def get_intersection_between_two_circles(p0, r0, p1, r1):
    """
    Returns the intersection of two circles on the WZ plane (X and Y projected onto W).
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)
    # Project x and y axes onto w axis.
    w0 = 0
    dw = math.sqrt(dx ** 2 + dy ** 2)
    # Calculate intersection points for both circles.
    # http://stackoverflow.com/a/3349134/798588
    d = math.sqrt(dw ** 2 + dz ** 2)
    # No intersections if circles cannot reach each other.
    if d > r0 + r1:
        return None
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r0 ** 2 - a ** 2)
    middle_w = w0 + a * dw / d
    middle_z = z0 + a * dz / d
    w2 = middle_w - h * dz / d
    z2 = middle_z + h * dw / d
    w3 = middle_w + h * dz / d
    z3 = middle_z - h * dw / d
    # Project w axis back onto x and y axes.
    x2 = x0 + w2 * dx / dw
    y2 = y0 + w2 * dy / dw
    x3 = x0 + w3 * dx / dw
    y3 = y0 + w3 * dy / dw
    return (x2, y2, z2), (x3, y3, z3)

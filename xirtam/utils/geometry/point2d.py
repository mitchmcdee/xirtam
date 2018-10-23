"""
Module containing a two-dimensional point class.
"""
from xirtam.utils.geometry.vector2d import Vector2D


class Point2D(Vector2D):
    """
    A two-dimensional point.
    """

    def __init__(self, x, y):
        super().__init__(x, y)

    def rotated_around_point(self, angle: float, point: "Point2D"):
        """
        Rotates this point counterclockwise by angle radians around the given point
        and returns the result.
        """
        return (self - point).rotated(angle) + point

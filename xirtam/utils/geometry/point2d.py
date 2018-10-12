"""
Module containing a two-dimensional point class.
"""
import vector2d


class Point2D(vector2d.Vector2D):
    """
    A two-dimensional point.
    """

    def __init__(self, x, y):
        super().__init__(x, y)

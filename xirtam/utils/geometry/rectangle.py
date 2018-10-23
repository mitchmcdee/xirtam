"""
Module containing a two-dimensional rectangle class.
"""
from typing import Any
from xirtam.utils.utils import clamp
from xirtam.utils.geometry.point2d import Point2D
import xirtam.utils.geometry.circle  # Necessary to avoid circular import.


class Rectangle:
    """
    A two-dimensional rectangle.
    Note: x and y represent the rectangle's bottom-left coordinate.
    """

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.left = x
        self.bottom = y
        self.right = x + width
        self.top = y + height
        self.centre = Point2D(self.x + self.width / 2, self.y + self.height / 2)
        self.corners = (
            Point2D(self.left, self.top),
            Point2D(self.right, self.top),
            Point2D(self.right, self.bottom),
            Point2D(self.left, self.bottom),
        )
        self.bounds = (self.left, self.top, self.right, self.bottom)
        self.vertices = [v for c in self.corners for v in c]

    def __eq__(self, other) -> bool:
        """
        Returns True if the rectangles are at the same position with the same dimension, else False.
        """
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
        )

    def __hash__(self):
        """
        Returns the unique hash for the rectangle.
        """
        return hash((self.x, self.y, self.width, self.height))

    def contains(self, other: Any) -> bool:
        """
        Returns True if the rectangle completely contains the other object, else False.
        """
        if isinstance(other, xirtam.utils.geometry.circle.Circle):
            return self.contains_circle(other)
        raise NotImplementedError()

    def contains_circle(self, circle: "xirtam.utils.geometry.circle.Circle") -> bool:
        """
        Returns True if the rectangle completely contains the other circle, else False.
        """
        return (
            self.left <= circle.left
            and circle.right <= self.right
            and self.bottom <= circle.bottom
            and circle.top <= self.top
        )

    def intersects(self, other: Any) -> bool:
        """
        Returns True if the other object intersects the rectangle, else False.
        """
        if isinstance(other, xirtam.utils.geometry.circle.Circle):
            return self.intersects_circle(other)
        elif isinstance(other, Point2D):
            return self.intersects_point2d(other)
        raise NotImplementedError()

    def intersects_circle(self, circle: "xirtam.utils.geometry.circle.Circle") -> bool:
        """
        Returns True if the other circle intersects the rectangle, else False.
        """
        # Calculate the distance between the circle's centre and this closest point
        dx = circle.x - clamp(circle.x, self.left, self.right)
        dy = circle.y - clamp(circle.y, self.bottom, self.top)
        # If the distance is less than the circle's radius, an intersection occurs
        return dx ** 2 + dy ** 2 <= circle.radius ** 2

    def intersects_point2d(self, point: Point2D) -> bool:
        """
        Returns True if the other point intersects the rectangle, else False.
        """
        return self.left <= point.x <= self.right and self.top >= point.y >= self.bottom

    def __repr__(self) -> str:
        """
        Returns the representation of the rectangle.
        """
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.width}, {self.height})"

"""
Module containing a two-dimensional circle class.
"""
import rectangle
from utils import clamp
from typing import Union, Any
from point2d import Point2D


class Circle:
    """
    A two-dimensional circle.
    """

    def __init__(self, centre: Union[Point2D, tuple], radius: float) -> None:
        if isinstance(centre, tuple):
            centre = Point2D(*centre)
        self.centre = centre
        self.radius = radius
        self.x = centre.x
        self.y = centre.y
        self.left = centre.x - radius
        self.right = centre.x + radius
        self.top = centre.y + radius
        self.bottom = centre.y - radius
        self.bounds = (self.left, self.top, self.right, self.bottom)

    def __eq__(self, other) -> bool:
        """
        Returns True if the circles are at the same position with the same radius, else False.
        """
        return self.centre == other.centre and self.radius == other.radius

    def __hash__(self):
        """
        Returns the unique hash for the circle.
        """
        return hash((self.centre, self.radius))

    def intersects(self, other: Any) -> bool:
        """
        Returns True if the other object intersects the circle, else False.
        """
        if isinstance(other, Circle):
            return self.intersects_circle(other)
        elif isinstance(other, rectangle.Rectangle):
            return self.intersects_rectangle(other)
        elif isinstance(other, Point2D):
            return self.intersects_point2d(other)
        raise NotImplementedError()

    def intersects_circle(self, circle: "Circle") -> bool:
        """
        Returns True if the other circle intersects the circle, else False.
        """
        return (self.radius + circle.radius) ** 2 > (self.x - circle.x) ** 2 + (
            self.y - circle.y
        ) ** 2

    def intersects_rectangle(self, rectangle: "Rectangle") -> bool:
        """
        Returns True if the other rectangle intersects the circle, else False.
        """
        # Calculate the distance between the circle's center and this closest point
        dx = self.x - clamp(self.x, rectangle.left, rectangle.right)
        dy = self.y - clamp(self.y, rectangle.bottom, rectangle.top)
        # If the distance is less than the circle's radius, an intersection occurs
        return dx ** 2 + dy ** 2 <= self.radius ** 2

    def intersects_point2d(self, point: "Point2D"):
        """
        Returns True if the other point intersects the circle, else False.
        """
        return (point.x - self.x) ** 2 + (point.y - self.y) ** 2 <= self.radius ** 2

    def __repr__(self) -> str:
        """
        Returns the representation of the circle.
        """
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.radius})"

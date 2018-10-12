"""
Module containing a two-dimensional vector class.
"""
import math


class Vector2D:
    """
    A two-dimensional vector.
    """

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.coords = (x, y)

    def __eq__(self, other) -> bool:
        """
        Returns True if the two vectors are of the same position, else False.
        """
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        """
        Returns the unique hash for the vector.
        """
        return hash((self.x, self.y))

    def __abs__(self) -> float:
        """
        Returns the absolute length of the vector.
        """
        return self.length

    def __neg__(self):
        """
        Returns the negation of the vector.
        """
        return self.__class__(-self.x, -self.y)

    def __add__(self, other):
        """
        Returns the addition between two vectors.
        """
        assert isinstance(other, self.__class__)
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """
        Returns the subtraction between two vectors.
        """
        assert isinstance(other, self.__class__)
        return self.__class__(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        """
        Returns the multiplication between two vectors.
        """
        assert isinstance(other, (int, float))
        return self.__class__(self.x * other, self.y * other)

    def __div__(self, other):
        """
        Returns the division between two vectors.
        """
        assert isinstance(other, (int, float))
        return self.__class__(self.x / other, self.y / other)

    @property
    def zero(self) -> bool:
        """
        Returns True if the vector is a zero-vector (i.e. at origin), else False.
        """
        return self.x == 0 and self.y == 0

    @property
    def normalized(self):
        """
        Returns the vector normalized.
        """
        if self.zero:
            return self.__class__(0, 0)
        else:
            return self.__class__(self.x / self.length, self.y / self.length)

    @property
    def angle(self) -> float:
        """
        Returns the angle of the vector.
        """
        if self.zero:
            raise ArithmeticError("Null vector has no angle")
        a = math.atan2(self.y, self.x)
        if a < 0:
            a += 2 * math.pi
        return a

    @property
    def length(self) -> float:
        """
        Returns the length of the vector.
        """
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def rotated(self, angle: float):
        """
        Rotates this vector counterclockwise by angle radians and returns the result.
        """
        if self.zero:
            return self.__class__(0, 0)
        else:
            return self.__class__.from_angle(self.angle + angle) * self.length

    def rotated_around_point(self, angle: float, point: "Point2D"):
        """
        Rotates this vector counterclockwise by angle radians around the given point
        and returns the result.
        """
        return (self - point).rotated(angle) + point

    def directed_angle_between(self, other: "Vector2D") -> float:
        """
        Returns a directed angle between two vectors.
        """
        return (other.angle - self.angle + math.pi) % (math.pi * 2) - math.pi

    def distance_to(self, other: "Vector2D") -> float:
        """
        Returns the distance to the other vector.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @classmethod
    def from_angle(cls, angle: float):
        """
        Returns the vector from the given angle.
        """
        assert isinstance(angle, (int, float))
        return cls(math.cos(angle), math.sin(angle))

    @classmethod
    def from_points(cls, point0: "Point2D", point1: "Point2D"):
        """
        Returns the vector from point0 to point1.
        """
        return point1 - point0

    def __iter__(self):
        """
        Returns the vector coordinates as an iterator for the vector.
        """
        yield self.x
        yield self.y

    def __repr__(self):
        """
        Returns the representation of the vector.
        """
        return f"{self.__class__.__name__}({self.x}, {self.y})"

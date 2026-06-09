"""
Single bounding box in XYXY (two corners) format.

The value array has length 4: x1, y1, x2, y2 in absolute pixel coordinates.
x1 must be strictly less than x2 and y1 strictly less than y2.
"""
import numpy as np
from ....array_types import FloatArray
from dataclasses import dataclass

from ..box import Box2D
from ..format import Box2DFormat


@dataclass
class Box2D_XYXY(Box2D):
    """
    XYXY format: top-left (x1, y1) and bottom-right (x2, y2) corners.

    Coordinate layout
    -----------------
    value has shape (4,) in row order:

        x1, y1, x2, y2

    x1, y1 are the top-left corner; x2, y2 are the bottom-right corner in
    image coordinates (x right, y down).

    Attributes
    ----------
    value : FloatArray
        Bounding box as x1, y1, x2, y2.

    Raises
    ------
    ValueError
        If x1 >= x2 or y1 >= y2 after construction.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.x1 >= self.x2:
            raise ValueError(
                f"x1 must be less than x2, but got x1={self.x1} and x2={self.x2}"
            )
        if self.y1 >= self.y2:
            raise ValueError(
                f"y1 must be less than y2, but got y1={self.y1} and y2={self.y2}"
            )

    @property
    def box_format(self) -> Box2DFormat:
        """
        Canonical format tag for this representation.

        Returns
        -------
        Box2DFormat
            The canonical format tag for this representation.
        """
        return Box2DFormat.XYXY

    @property
    def width(self) -> float:
        """
        Width of the box (horizontal extent).

        Returns
        -------
        float
            x2 minus x1 in pixels.
        """
        return self.value[2] - self.value[0]

    @property
    def height(self) -> float:
        """
        Height of the box (vertical extent).

        Returns
        -------
        float
            y2 minus y1 in pixels.
        """
        return self.value[3] - self.value[1]

    @property
    def x1(self) -> float:
        """
        Left edge (minimum x) of the box.

        Returns
        -------
        float
            Index 0 of value — left x in pixel coordinates.
        """
        return self.value[0]

    @property
    def y1(self) -> float:
        """
        Top edge (minimum y) of the box.

        Returns
        -------
        float
            Index 1 of value — top y in pixel coordinates.
        """
        return self.value[1]

    @property
    def x2(self) -> float:
        """
        Right edge (maximum x) of the box.

        Returns
        -------
        float
            Index 2 of value — right x in pixel coordinates.
        """
        return self.value[2]

    @property
    def y2(self) -> float:
        """
        Bottom edge (maximum y) of the box.

        Returns
        -------
        float
            Index 3 of value — bottom y in pixel coordinates.
        """
        return self.value[3]

    @property
    def y_max(self) -> float:
        """
        Maximum y-coordinate of the box (same as y2 for XYXY).

        Returns
        -------
        float
            Bottom y of the rectangle.
        """
        return self.value[3]

    @property
    def area(self) -> float:
        """
        Area of the axis-aligned rectangle.

        Returns
        -------
        float
            (x2 - x1) times (y2 - y1).
        """
        return (self.value[2] - self.value[0]) * (self.value[3] - self.value[1])

    @property
    def center(self) -> FloatArray:
        """
        Center (cx, cy) of the bounding box.

        Returns
        -------
        FloatArray
            Shape (2,): midpoint of the diagonal between (x1, y1) and (x2, y2).
        """
        x_min, y_min, x_max, y_max = self.value
        return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

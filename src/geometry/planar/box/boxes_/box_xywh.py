"""
Single bounding box in XYWH (top-left + size) format.

The value array has length 4: x_min, y_min, width, height in absolute pixel
coordinates. Width and height must be strictly positive.
"""
import numpy as np
from ....array_types import FloatArray
from dataclasses import dataclass

from ..box import Box2D
from ..format import Box2DFormat


@dataclass
class Box2D_XYWH(Box2D):
    """
    XYWH format: top-left (x_min, y_min) plus width and height.

    Coordinate layout
    -----------------
    value has shape (4,) in row order:

        x_min, y_min, width, height

    x_min and y_min are the top-left corner in image coordinates (x right,
    y down). width and height extend to the right and down.

    Attributes
    ----------
    value : FloatArray
        Bounding box as x_min, y_min, width, height.

    Raises
    ------
    ValueError
        If width <= 0 or height <= 0 after construction.
    """

    def __post_init__(self):
        # Base class checks shape (4,); here we require positive width and height.
        super().__post_init__()
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"width and height must be greater than 0, but got width={self.width} and height={self.height}"
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
        return Box2DFormat.XYWH

    @property
    def width(self) -> float:
        """
        Width of the box (horizontal extent).

        Returns
        -------
        float
            Index 2 of value — width in pixels.
        """
        return self.value[2]

    @property
    def height(self) -> float:
        """
        Height of the box (vertical extent).

        Returns
        -------
        float
            Index 3 of value — height in pixels.
        """
        return self.value[3]

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
    def x2(self) -> float:
        """
        Right edge: x_min + width.

        Returns
        -------
        float
            Sum of left x and width.
        """
        return self.value[0] + self.value[2]

    @property
    def y2(self) -> float:
        """
        Bottom edge: y_min + height.

        Returns
        -------
        float
            Sum of top y and height.
        """
        return self.value[1] + self.value[3]

    @property
    def y_max(self) -> float:
        """
        Maximum y-coordinate of the box (same as y2 for XYWH).

        Returns
        -------
        float
            Bottom y of the rectangle.
        """
        return self.y2

    @property
    def area(self) -> float:
        """
        Area of the axis-aligned rectangle.

        Returns
        -------
        float
            width times height.
        """
        return self.value[2] * self.value[3]

    @property
    def center(self) -> FloatArray:
        """
        Center (cx, cy) of the bounding box.

        Returns
        -------
        FloatArray
            Shape (2,): x_min + w/2, y_min + h/2.
        """
        x_min, y_min, w, h = self.value
        return np.array([x_min + w / 2, y_min + h / 2])

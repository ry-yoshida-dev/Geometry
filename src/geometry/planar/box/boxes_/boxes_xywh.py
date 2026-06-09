"""
Batch of bounding boxes in XYWH (top-left + size) format.

Each row has length 4: x_min, y_min, width, height in absolute pixel
coordinates. Width and height must be strictly positive for every box.
"""
import numpy as np
from ....array_types import FloatArray
from dataclasses import dataclass

from ..boxes import Boxes2D
from ..format import Box2DFormat


@dataclass
class Boxes2D_XYWH(Boxes2D):
    """
    XYWH format: top-left (x_min, y_min) plus width and height per row.

    Coordinate layout
    -----------------
    value has shape (N, 4). Each row is:

        x_min, y_min, width, height

    x_min and y_min are the top-left corner in image coordinates (x right,
    y down). width and height extend to the right and down.

    Attributes
    ----------
    value : FloatArray
        Bounding boxes as x_min, y_min, width, height per row.

    Raises
    ------
    ValueError
        If any width <= 0 or height <= 0 after construction.
    """

    def __post_init__(self):
        super().__post_init__()
        if np.any(self.width <= 0) or np.any(self.height <= 0):
            raise ValueError(
                "width and height must be greater than 0 for all boxes"
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
    def width(self) -> FloatArray:
        """
        Width of each box (horizontal extent).

        Returns
        -------
        FloatArray
            Shape (N,) — column 2 of value.
        """
        return self.value[:, 2]

    @property
    def height(self) -> FloatArray:
        """
        Height of each box (vertical extent).

        Returns
        -------
        FloatArray
            Shape (N,) — column 3 of value.
        """
        return self.value[:, 3]

    @property
    def x1(self) -> FloatArray:
        """
        Left edge (minimum x) of each box.

        Returns
        -------
        FloatArray
            Shape (N,) — column 0 of value.
        """
        return self.value[:, 0]

    @property
    def y1(self) -> FloatArray:
        """
        Top edge (minimum y) of each box.

        Returns
        -------
        FloatArray
            Shape (N,) — column 1 of value.
        """
        return self.value[:, 1]

    @property
    def x2(self) -> FloatArray:
        """
        Right edge of each box: x_min + width.

        Returns
        -------
        FloatArray
            Shape (N,) — left x plus width per row.
        """
        return self.value[:, 0] + self.value[:, 2]

    @property
    def y2(self) -> FloatArray:
        """
        Bottom edge of each box: y_min + height.

        Returns
        -------
        FloatArray
            Shape (N,) — top y plus height per row.
        """
        return self.value[:, 1] + self.value[:, 3]

    @property
    def y_max(self) -> FloatArray:
        """
        Maximum y-coordinate of each box (same as y2 for XYWH).

        Returns
        -------
        FloatArray
            Shape (N,) — bottom y per box.
        """
        return self.y2

    @property
    def area(self) -> FloatArray:
        """
        Area of each axis-aligned rectangle.

        Returns
        -------
        FloatArray
            Shape (N,) — width times height per row.
        """
        return self.value[:, 2] * self.value[:, 3]

    @property
    def center(self) -> FloatArray:
        """
        Center (cx, cy) of each bounding box.

        Returns
        -------
        FloatArray
            Shape (N, 2): x_min + w/2, y_min + h/2 per row.
        """
        x_min = self.value[:, 0]
        y_min = self.value[:, 1]
        w = self.value[:, 2]
        h = self.value[:, 3]
        return np.stack([x_min + w / 2, y_min + h / 2], axis=-1)

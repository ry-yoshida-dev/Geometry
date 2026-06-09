"""
Batch of bounding boxes in XYXY (two corners) format.

Each row has length 4: x1, y1, x2, y2 in absolute pixel coordinates.
For every box, x1 must be strictly less than x2 and y1 strictly less than y2.
"""
import numpy as np
from ....array_types import NumericArray
from dataclasses import dataclass

from ..boxes import Boxes2D
from ..format import Box2DFormat


@dataclass
class Boxes2D_XYXY(Boxes2D):
    """
    XYXY format: top-left (x1, y1) and bottom-right (x2, y2) per row.

    Coordinate layout
    -----------------
    value has shape (N, 4). Each row is:

        x1, y1, x2, y2

    x1, y1 are the top-left corner; x2, y2 are the bottom-right corner in
    image coordinates (x right, y down).

    Attributes
    ----------
    value : NumericArray
        Bounding boxes as x1, y1, x2, y2 per row.

    Raises
    ------
    ValueError
        If any row has x1 >= x2 or y1 >= y2 after construction.
    """

    def __post_init__(self):
        super().__post_init__()
        if np.any(self.x1 >= self.x2) or np.any(self.y1 >= self.y2):
            raise ValueError(
                "x1 must be less than x2 and y1 less than y2 for all boxes"
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
    def width(self) -> NumericArray:
        """
        Width of each box (horizontal extent).

        Returns
        -------
        NumericArray
            Shape (N,) — x2 minus x1 per row.
        """
        return self.value[:, 2] - self.value[:, 0]

    @property
    def height(self) -> NumericArray:
        """
        Height of each box (vertical extent).

        Returns
        -------
        NumericArray
            Shape (N,) — y2 minus y1 per row.
        """
        return self.value[:, 3] - self.value[:, 1]

    @property
    def x1(self) -> NumericArray:
        """
        Left edge (minimum x) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) — column 0 of value.
        """
        return self.value[:, 0]

    @property
    def y1(self) -> NumericArray:
        """
        Top edge (minimum y) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) — column 1 of value.
        """
        return self.value[:, 1]

    @property
    def x2(self) -> NumericArray:
        """
        Right edge (maximum x) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) — column 2 of value.
        """
        return self.value[:, 2]

    @property
    def y2(self) -> NumericArray:
        """
        Bottom edge (maximum y) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) — column 3 of value.
        """
        return self.value[:, 3]

    @property
    def y_max(self) -> NumericArray:
        """
        Maximum y-coordinate of each box (same as y2 for XYXY).

        Returns
        -------
        NumericArray
            Shape (N,) — bottom y per box.
        """
        return self.value[:, 3]

    @property
    def area(self) -> NumericArray:
        """
        Area of each axis-aligned rectangle.

        Returns
        -------
        NumericArray
            Shape (N,) — (x2 - x1) times (y2 - y1) per row.
        """
        return (self.value[:, 2] - self.value[:, 0]) * (
            self.value[:, 3] - self.value[:, 1]
        )

    @property
    def center(self) -> NumericArray:
        """
        Center (cx, cy) of each bounding box.

        Returns
        -------
        NumericArray
            Shape (N, 2): midpoint of the diagonal per row.
        """
        x_min, y_min, x_max, y_max = (
            self.value[:, 0],
            self.value[:, 1],
            self.value[:, 2],
            self.value[:, 3],
        )
        return np.stack([(x_min + x_max) / 2, (y_min + y_max) / 2], axis=-1)

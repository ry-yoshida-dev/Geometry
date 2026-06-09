"""
Single 2D bounding box API extending the abstract base in base.

Concrete formats (XYXY, XYWH) are implemented in boxes_. This module
validates value with shape (4,) and provides conversion, cropping,
and geometry helpers.
"""
from __future__ import annotations
from ...array_types import NumericArray
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from shapely.geometry import Polygon

from .base import Box2D as Box2DBase
from .format import Box2DFormat
from .utils import Box2dConverter

@dataclass
class Box2D(Box2DBase[float, tuple[slice, slice]]):
    """
    Abstract single bounding box with value of shape (4,).

    Subclasses define the coordinate layout (e.g. XYXY or XYWH). Coordinates
    are absolute pixel values in image space (x right, y down).

    Attributes
    ----------
    value : NumericArray
        Length-4 array; interpretation depends on :attr:`box_format`.
    """

    def __post_init__(self):
        """
        Validate shape after initialization.

        Raises
        ------
        ValueError
            If value is not shape (4,).
        """
        if self.value.shape != (4,):
            raise ValueError(f"box must have shape (4,), but got {self.value.shape}")

    @property
    @abstractmethod
    def box_format(self) -> Box2DFormat:
        """
        Canonical format tag for this representation.

        Returns
        -------
        Box2DFormat
            The canonical format tag for this representation.
        """

    @property
    @abstractmethod
    def width(self) -> float:
        """
        Width of the box (horizontal extent).

        Returns
        -------
        float
            Width in pixels.
        """

    @property
    @abstractmethod
    def height(self) -> float:
        """
        Height of the box (vertical extent).

        Returns
        -------
        float
            Height in pixels.
        """

    @property
    @abstractmethod
    def x1(self) -> float:
        """
        Left edge (minimum x) of the box.

        Returns
        -------
        float
            Minimum x in pixel coordinates.
        """

    @property
    @abstractmethod
    def y1(self) -> float:
        """
        Top edge (minimum y) of the box.

        Returns
        -------
        float
            Minimum y in pixel coordinates.
        """

    @property
    @abstractmethod
    def x2(self) -> float:
        """
        Right edge (maximum x) of the box.

        Returns
        -------
        float
            Maximum x in pixel coordinates.
        """

    @property
    @abstractmethod
    def y2(self) -> float:
        """
        Bottom edge (maximum y) of the box.

        Returns
        -------
        float
            Maximum y in pixel coordinates.
        """
    
    @property
    @abstractmethod
    def y_max(self) -> float:
        """
        Maximum y-coordinate of the box (often same as y2).

        Returns
        -------
        float
            Bottom y of the rectangle in pixel coordinates.
        """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Area of the axis-aligned rectangle.

        Returns
        -------
        float
            Area in square pixels.
        """
        pass

    @property
    @abstractmethod
    def center(self) -> NumericArray:
        """
        Center (cx, cy) of the bounding box.

        Returns
        -------
        NumericArray
            Shape (2,) - center in pixel coordinates.
        """

    def to_format(
        self, 
        target_format: Box2DFormat, 
        ) -> NumericArray:
        """
        Convert this box to another coordinate layout.

        Parameters
        ----------
        target_format : Box2DFormat
            Desired output format (e.g. XYXY or XYWH).

        Returns
        -------
        NumericArray
            Shape (4,) - same box in target_format.
        """
        return Box2dConverter.convert_format(
            self.value.reshape(1, 4),
            input_format=self.box_format,
            output_format=target_format,
            as_int=False
            ).reshape(4,)

    @property
    def crop_slice(self) -> tuple[slice, slice]:
        """
        Integer slice pair for cropping a 2D image with NumPy indexing.

        Equivalent to image[yslice, xslice] when image is indexed
        (row, col) as (y, x).

        Returns
        -------
        tuple[slice, slice]
            (yslice, xslice) from rounded box edges in pixel coordinates.
        """
        y_min = int(round(self.y1))
        y_max = int(round(self.y2))
        x_min = int(round(self.x1))
        x_max = int(round(self.x2))

        return (
            slice(y_min, y_max),
            slice(x_min, x_max)
            )

    @property
    def aspect_ratio(self) -> float:
        """
        Aspect ratio height / width of the bounding box.

        Returns
        -------
        float
            Height divided by width.

        Raises
        ------
        ZeroDivisionError
            If width is zero.
        """
        if self.width == 0:
            raise ZeroDivisionError("Width of the box is zero, cannot compute aspect ratio.")
        return self.height / self.width

    @property
    def shapely(self) -> Polygon:
        """
        Axis-aligned rectangle as a Shapely polygon.

        Returns
        -------
        Polygon
            Closed ring with four corners from XYXY corners.
        """
        x1, y1, x2, y2 = self.to_format(Box2DFormat.XYXY)
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        box_str = np.array2string(self.value, precision=2, separator=', ')
        return (
            f"{cls_name}(value={box_str}, "
            f"box_format='{self.box_format}')"
        )

    @classmethod
    def register(
        cls, 
        value: NumericArray, 
        box2d_format: Box2DFormat
        ) -> Box2D:
        """
        Construct the appropriate concrete :class:`Box2D` for box2d_format.

        Parameters
        ----------
        value : NumericArray
            Shape (4,) - layout depends on box2d_format.
        box2d_format : Box2DFormat
            Input coordinate convention (XYXY, XYWH, or aliases).

        Returns
        -------
        Box2D
            Concrete subclass instance wrapping value.
        """
        match box2d_format:
            case Box2DFormat.XYXY | Box2DFormat.TLBR:
                from .boxes_ import Box2D_XYXY
                cls_ = Box2D_XYXY
            case Box2DFormat.XYWH | Box2DFormat.TLWH:
                from .boxes_ import Box2D_XYWH
                cls_ = Box2D_XYWH
            case _:
                raise ValueError(f"Unsupported box format: {box2d_format}")
        return cls_(value=value)


if __name__ == "__main__":
    value = np.array([100, 100, 200, 200])
    box2d_format = Box2DFormat.XYXY
    box2d = Box2D.register(
        value=value, 
        box2d_format=box2d_format
        )
    print(box2d)

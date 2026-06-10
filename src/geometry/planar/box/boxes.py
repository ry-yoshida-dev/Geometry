"""
Batch of 2D bounding boxes (N rows, 4 columns) extending base.Box2D.

Concrete batched formats (XYXY, XYWH) live in boxes_. This module validates
shape (N, 4) and provides per-box conversion, cropping, and Shapely access.
"""
from __future__ import annotations
from ...array_types import FloatArray, NumericArray
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from shapely.geometry import Polygon

from .base import Box2D as Box2DBase
from .box import Box2D
from .format import Box2DFormat
from .utils import Box2dConverter

@dataclass
class Boxes2D(Box2DBase[NumericArray, list[tuple[slice, slice]]]):
    """
    Abstract stack of N bounding boxes with value of shape (N, 4).

    Each row uses the same coordinate convention as the concrete subclass
    (e.g. XYXY or XYWH). Coordinates are absolute pixel values.

    Attributes
    ----------
    value : NumericArray
        Shape (N, 4); row layout depends on :attr:`box_format`.
    """

    def __post_init__(self):
        """
        Validate shape after initialization and coerce to ndarray.

        Raises
        ------
        ValueError
            If value is not 2-D with second dimension 4.
        """
        v = np.asarray(self.value)
        if v.ndim != 2 or v.shape[1] != 4:
            raise ValueError(
                f"boxes must have shape (N, 4), but got {v.shape}"
            )
        self.value = v

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
    def width(self) -> NumericArray:
        """
        Width of each box (horizontal extent).

        Returns
        -------
        NumericArray
            Shape (N,) - width in pixels per row.
        """

    @property
    @abstractmethod
    def height(self) -> NumericArray:
        """
        Height of each box (vertical extent).

        Returns
        -------
        NumericArray
            Shape (N,) - height in pixels per row.
        """

    @property
    @abstractmethod
    def x1(self) -> NumericArray:
        """
        Left edge (minimum x) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) - minimum x per row.
        """

    @property
    @abstractmethod
    def y1(self) -> NumericArray:
        """
        Top edge (minimum y) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) - minimum y per row.
        """

    @property
    @abstractmethod
    def x2(self) -> NumericArray:
        """
        Right edge (maximum x) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) - maximum x per row.
        """

    @property
    @abstractmethod
    def y2(self) -> NumericArray:
        """
        Bottom edge (maximum y) of each box.

        Returns
        -------
        NumericArray
            Shape (N,) - maximum y per row.
        """
    
    @property
    @abstractmethod
    def y_max(self) -> NumericArray:
        """
        Maximum y-coordinate of each box (often same as y2).

        Returns
        -------
        NumericArray
            Shape (N,) - bottom y per box.
        """

    @property
    @abstractmethod
    def area(self) -> NumericArray:
        """
        Area of each axis-aligned rectangle.

        Returns
        -------
        NumericArray
            Shape (N,) - area in square pixels per row.
        """

    @property
    @abstractmethod
    def center(self) -> NumericArray:
        """
        Center (cx, cy) of each bounding box.

        Returns
        -------
        NumericArray
            Shape (N, 2) - centers in pixel coordinates.
        """

    def to_format(
        self, 
        target_format: Box2DFormat, 
        ) -> NumericArray:
        """
        Convert all rows to another coordinate layout.

        Parameters
        ----------
        target_format : Box2DFormat
            Desired output format (e.g. XYXY or XYWH).

        Returns
        -------
        NumericArray
            Shape (N, 4) - same boxes in target_format.
        """
        return Box2dConverter.convert_format(
            self.value,
            input_format=self.box_format,
            output_format=target_format,
            as_int=False
            )

    @property
    def crop_slice(self) -> list[tuple[slice, slice]]:
        """
        Integer slice pairs for cropping a 2D image per box.

        Returns
        -------
        list[tuple[slice, slice]]
            Length N; each element is (yslice, xslice) for NumPy
            image[yslice, xslice] indexing.
        """
        xyxy = self.to_format(Box2DFormat.XYXY)
        out: list[tuple[slice, slice]] = []
        for row in xyxy:
            x1, y1, x2, y2 = row
            y_min = int(round(y1))
            y_max = int(round(y2))
            x_min = int(round(x1))
            x_max = int(round(x2))
            out.append((slice(y_min, y_max), slice(x_min, x_max)))
        return out

    @property
    def aspect_ratio(self) -> FloatArray:
        """
        Aspect ratio height / width for each bounding box.

        Returns
        -------
        FloatArray
            Shape (N,).

        Raises
        ------
        ZeroDivisionError
            If any box has zero width.
        """
        if np.any(self.width == 0):
            raise ZeroDivisionError("Width of at least one box is zero, cannot compute aspect ratio.")
        return np.divide(self.height, self.width, dtype=np.float64)

    @property
    def shapely(self) -> list[Polygon]:
        """
        Axis-aligned rectangle for each box as a Shapely polygon.

        Returns
        -------
        list[Polygon]
            Length N - one closed ring per row.
        """
        xyxy = self.to_format(Box2DFormat.XYXY)
        polys: list[Polygon] = []
        for row in xyxy:
            x1, y1, x2, y2 = row
            polys.append(
                Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            )
        return polys

    def __len__(self) -> int:
        """
        Number of bounding boxes in the batch.

        Returns
        -------
        int
            Row count N of :attr:`value`.
        """
        return self.value.shape[0]

    def __getitem__(self, index: int | slice) -> Box2D | Boxes2D:
        """
        Index or slice rows of the batch.

        Parameters
        ----------
        index : int | slice
            Row index for one :class:`Box2D`, or slice for a sub-batch.

        Returns
        -------
        Box2D | Boxes2D
            One box or a new batch of the same concrete subclass.
        """
        result = self.value[index]
        if isinstance(index, slice):
            return type(self)(value=result)
        return Box2D.register(value=result, box2d_format=self.box_format)

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
        ) -> Boxes2D:
        """
        Construct the appropriate concrete :class:`Boxes2D` for box2d_format.

        Parameters
        ----------
        value : NumericArray
            Shape (N, 4) - layout depends on box2d_format.
        box2d_format : Box2DFormat
            Row coordinate convention (XYXY, XYWH, or aliases).

        Returns
        -------
        Boxes2D
            Concrete subclass instance wrapping value.
        """
        match box2d_format:
            case Box2DFormat.XYXY | Box2DFormat.TLBR:
                from .boxes_ import Boxes2D_XYXY
                cls_ = Boxes2D_XYXY
            case Box2DFormat.XYWH | Box2DFormat.TLWH:
                from .boxes_ import Boxes2D_XYWH
                cls_ = Boxes2D_XYWH
            case _:
                raise ValueError(f"Unsupported box format: {box2d_format}")
        return cls_(value=value)

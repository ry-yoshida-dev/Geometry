"""
Abstract 2D bounding box protocol for scalar or vector coordinate fields.

``T`` is either ``float`` (single box) or ``np.ndarray`` (batch).
Coordinates are always absolute pixel values in image space (x right, y down).
"""
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from .format import Box2DFormat
from .utils import Box2dConverter

T = TypeVar('T', float, np.ndarray)
CropSliceT = TypeVar(
    'CropSliceT',
    tuple[slice, slice],
    list[tuple[slice, slice]],
)

@dataclass
class Box2D(ABC, Generic[T, CropSliceT]):
    """
    Abstract 2D bounding box stored as a NumPy array.

    Subclasses fix the row layout (e.g. XYXY or XYWH). All coordinates are
    absolute pixel coordinates.

    Attributes
    ----------
    value : np.ndarray
        For a single box, shape ``(4,)``; for a batch, shape ``(N, 4)``.
    """
    value: np.ndarray

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
    def width(self) -> T:
        """
        Width of the box or each box (horizontal extent).

        Returns
        -------
        T
            Width in pixels (scalar or per-row array).
        """

    @property
    @abstractmethod
    def height(self) -> T:
        """
        Height of the box or each box (vertical extent).

        Returns
        -------
        T
            Height in pixels (scalar or per-row array).
        """

    @property
    @abstractmethod
    def x1(self) -> T:
        """
        Left edge (minimum x) of the box or each box.

        Returns
        -------
        T
            Minimum x in pixel coordinates.
        """

    @property
    @abstractmethod
    def y1(self) -> T:
        """
        Top edge (minimum y) of the box or each box.

        Returns
        -------
        T
            Minimum y in pixel coordinates.
        """

    @property
    @abstractmethod
    def x2(self) -> T:
        """
        Right edge (maximum x) of the box or each box.

        Returns
        -------
        T
            Maximum x in pixel coordinates.
        """

    @property
    @abstractmethod
    def y2(self) -> T:
        """
        Bottom edge (maximum y) of the box or each box.

        Returns
        -------
        T
            Maximum y in pixel coordinates.
        """
    
    @property
    @abstractmethod
    def y_max(self) -> T:
        """
        Maximum y-coordinate of the box or each box.

        Returns
        -------
        T
            Bottom y in pixel coordinates (often same as ``y2``).
        """

    @property
    @abstractmethod
    def area(self) -> T:
        """
        Area of the axis-aligned rectangle (each box if batched).

        Returns
        -------
        T
            Area in square pixels.
        """

    @property
    @abstractmethod
    def center(self) -> T:
        """
        Center ``(cx, cy)`` of the box or each box.

        Returns
        -------
        T
            For a batch, typically shape ``(N, 2)``.
        """

    @property
    @abstractmethod
    def crop_slice(self) -> CropSliceT:
        """
        Integer slice pair(s) for cropping a 2D image with NumPy indexing.

        For a single box, ``(yslice, xslice)`` for ``image[yslice, xslice]``.
        For a batch, one such pair per row (length ``N``).

        Returns
        -------
        CropSliceT
            ``tuple[slice, slice]`` or ``list[tuple[slice, slice]]``.
        """

    def to_format(
        self, 
        target_format: Box2DFormat, 
        ) -> np.ndarray:
        """
        Convert stored coordinates to another layout.

        Parameters
        ----------
        target_format : Box2DFormat
            Desired output format (e.g. XYXY or XYWH).

        Returns
        -------
        np.ndarray
            Same leading shape as ``value`` with last dimension 4.
        """
        return Box2dConverter.convert_format(
            self.value,
            input_format=self.box_format,
            output_format=target_format,
            as_int=False
            )

    @property
    @abstractmethod
    def aspect_ratio(self) -> T:
        """
        Aspect ratio ``height / width`` of the box or each box.

        Returns
        -------
        T
            Height divided by width.

        Raises
        ------
        ZeroDivisionError
            If any width is zero (batch: if any row has zero width).
        """
        if self.width == 0:
            raise ZeroDivisionError("Width of the box is zero, cannot compute aspect ratio.")
        return self.height / self.width

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        box_str = np.array2string(self.value, precision=2, separator=', ')
        return (
            f"{cls_name}(value={box_str}, "
            f"box_format='{self.box_format}')"
        )

    @classmethod
    @abstractmethod
    def register(
        cls, 
        value: np.ndarray, 
        box2d_format: Box2DFormat
        ) -> (
            Box2D[float, tuple[slice, slice]]
            | Box2D[np.ndarray, list[tuple[slice, slice]]]
        ):
        """
        Construct a concrete subclass from raw array data.

        Parameters
        ----------
        value : np.ndarray
            Coordinate array; shape ``(4,)`` or ``(N, 4)`` depending on subclass.
        box2d_format : Box2DFormat
            Layout of ``value`` (XYXY, XYWH, or aliases).

        Returns
        -------
        Box2D
            Concrete implementation instance.
        """



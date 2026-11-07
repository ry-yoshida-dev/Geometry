from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ...array_types import NumericArray
from ..point import Points3D
from ..vector import Vectors3D
from .base import Line3D as Line3DBase
from .line import Line3D


@dataclass
class Lines3D(Line3DBase[NumericArray]):
    """
    A batch of 3D line segments stored in an (N, 2, 3) array.
    """

    value: NumericArray

    def __post_init__(self) -> None:
        v = np.asarray(self.value)
        if v.ndim != 3 or v.shape[1:] != (2, 3):
            raise ValueError(f"lines must have shape (N, 2, 3), but got {v.shape}")
        self.value = v

    @property
    def start(self) -> Points3D:
        """
        Get the start points of the lines.

        Returns
        -------
        Points3D
            The start point of each line, with shape (N, 3).
        """
        return Points3D(value=self.value[:, 0, :])

    @property
    def end(self) -> Points3D:
        """
        Get the end points of the lines.

        Returns
        -------
        Points3D
            The end point of each line, with shape (N, 3).
        """
        return Points3D(value=self.value[:, 1, :])

    @property
    def length(self) -> NumericArray:
        """
        Get the Euclidean length of each line.

        Returns
        -------
        NumericArray
            The lengths, with shape (N,).
        """
        d = self.value[:, 1, :] - self.value[:, 0, :]
        return cast(NumericArray, np.linalg.norm(d, axis=1))

    @property
    def center(self) -> Points3D:
        """
        Get the midpoint of each line.

        Returns
        -------
        Points3D
            The midpoints, with shape (N, 3).
        """
        return Points3D(value=np.mean(self.value, axis=1))

    @property
    def vectors(self) -> Vectors3D:
        """
        Get the vector from the start point to the end point of each line.

        Returns
        -------
        Vectors3D
            The vectors, with shape (N, 3).
        """
        return Vectors3D(value=self.value[:, 1, :] - self.value[:, 0, :])

    def __len__(self) -> int:
        return int(self.value.shape[0])

    def __getitem__(self, index: int | slice) -> Line3D | Lines3D:
        result = self.value[index]
        if isinstance(index, slice):
            return Lines3D(value=result)
        return Line3D(value=result)

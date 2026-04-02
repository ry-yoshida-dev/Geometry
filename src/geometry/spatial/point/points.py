from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Iterator, Union
from scipy.spatial.distance import cdist

from .base import Point3D as Point3DBase
from .point import Point3D

@dataclass
class Points3D(Point3DBase[np.ndarray]):
    value: np.ndarray

    def __post_init__(self):
        if self.value.ndim != 2:
            raise ValueError(f"value must have shape (n, 3), got shape {self.value.shape}")
        if self.value.shape[1] != 3:
            raise ValueError(f"value must have 3 columns, got shape {self.value.shape}")

    @property
    def x(self) -> np.ndarray:
        return self.value[:, 0]
    
    @property
    def y(self) -> np.ndarray:
        return self.value[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.value[:, 2]

    @property
    def distance_matrix(self) -> np.ndarray:
        """
        Get the distance matrix of the points.

        Returns
        -------
        np.ndarray: The distance matrix of the points with shape (n, n).
        """
        return cdist(self.value, self.value, metric='euclidean')

    def scale(
        self, 
        scale: float
        ) -> Points3D:
        """
        Scale the points.

        Parameters
        ----------
        scale: float
            The scale factor.

        Returns
        -------
        Points3D: The scaled points.
        """
        return Points3D(value=self.value * scale)

    def __add__(
        self, 
        other: Points3D
        ) -> Points3D:
        """
        Add two points.

        Parameters
        ----------
        other: Points3D
            The other points.
        """
        return Points3D(value=self.value + other.value)

    def __sub__(
        self, 
        other: Points3D
        ) -> Points3D:
        """
        Subtract two points.

        Parameters
        ----------
        other: Points3D
            The other points.
        """
        return Points3D(value=self.value - other.value)

    def __mul__(
        self, 
        other: float
        ) -> Points3D:
        """
        Multiply the points by a scalar.

        Parameters
        ----------
        other: float
            The scalar.

        Returns
        -------
        Points3D: The multiplied points.
        """
        return Points3D(value=self.value * other)

    def __truediv__(
        self, 
        other: float
        ) -> Points3D:
        """
        Divide the points by a scalar.

        Parameters
        ----------
        other: float
            The scalar.

        Returns
        -------
        Points3D: The divided points.
        """
        return Points3D(value=self.value / other)

    def __repr__(self) -> str:
        return f"Points3D(n={len(self)}, shape={self.value.shape})"
    
    def __len__(self) -> int:
        return len(self.value)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Point3D, "Points3D"]:
        result = self.value[index]
        if isinstance(index, slice):
            return Points3D(value=result)
        return Point3D(value=result)
    
    def __iter__(self) -> Iterator[Point3D]:
        for value in self.value:
            yield Point3D(value=value)

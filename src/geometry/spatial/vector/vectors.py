from __future__ import annotations

from ...array_types import FloatArray
import numpy as np
from dataclasses import dataclass
from typing import Iterator
from .vector import Vector3D
from units import (
    Angle, 
    AngleUnit
    )

@dataclass
class Vectors3D:
    value: FloatArray

    def __post_init__(self):
        if self.value.ndim != 2:
            raise ValueError(f"value must have shape (n, 3), got shape {self.value.shape}")
        if self.value.shape[1] != 3:
            raise ValueError(f"value must have shape (n, 3), got shape {self.value.shape}")

    @property
    def x(self) -> FloatArray:
        return self.value[:, 0]

    @property
    def y(self) -> FloatArray:
        return self.value[:, 1]
    
    @property
    def z(self) -> FloatArray:
        return self.value[:, 2]
    
    @property
    def unit_vectors(self) -> FloatArray:
        """
        Return the unit vectors of the vectors.
        
        Returns
        -------
        FloatArray:
            The unit vectors of the vectors in the form of (n, 3).
        """
        norm = np.linalg.norm(self.value, axis=-1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("Cannot compute a unit vector from a zero-length vector.")
        return self.value / norm

    def to_azimuthal_angles(
        self, 
        up_index: int
        ) -> Angle:
        """
        Convert the points to azimuthal angles.
        
        Parameters
        ----------
        up_index: int
            The index of the up axis.

        Returns
        -------
        Angle:
            The azimuthal angles.
        """
        if up_index not in [0, 1, 2]:
            raise ValueError(f"Invalid up index: {up_index}")
        azimuthal_angles = np.arccos(np.clip(self.unit_vectors[..., up_index], -1, 1)).flatten()
        return Angle(
            value=azimuthal_angles, 
            unit=AngleUnit.RADIAN
            )

    def to_polar_angles(
        self, 
        forward_index: int, 
        right_index: int
        ) -> Angle:
        """
        Convert the points to polar angles.
        
        Parameters
        ----------
        forward_index: int
            The index of the forward axis.
        right_index: int
            The index of the right axis.
            
        Returns
        -------
        Angle:
            The polar angles.
        """
        if forward_index not in [0, 1, 2]:
            raise ValueError(f"Invalid forward index: {forward_index}")
        if right_index not in [0, 1, 2]:
            raise ValueError(f"Invalid right index: {right_index}")
        if forward_index == right_index:
            raise ValueError(f"Forward index and right index must be different")
        unit_vectors = self.unit_vectors
        polar_angles = np.arctan2(
            unit_vectors[..., right_index], 
            unit_vectors[..., forward_index]
            ).flatten()
        return Angle(
            value=polar_angles, 
            unit=AngleUnit.RADIAN
            )

    def __len__(self) -> int:
        """
        Return the number of vectors.
        
        Returns
        -------
        int:
            The number of vectors.
        """
        return len(self.value)

    def __getitem__(
        self, 
        index: int
        ) -> Vector3D:
        """
        Return the vector at the given index.
        
        Returns
        -------
        Vector3D:
            The vector at the given index.
        """
        return Vector3D(value=self.value[index])

    def __iter__(self) -> Iterator[Vector3D]:
        for value in self.value:
            yield Vector3D(value=value)
    
    def __str__(self) -> str:
        return f"Vectors3D(n={len(self)}, shape={self.value.shape})"

    

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .normal_value import NormalValue
from cartesian_axis import CartesianCoordinateSystem


@dataclass
class NormalMap:
    """
    A class representing a normal map.
    
    Parameters
    ----------
    value: np.ndarray
        The normal map represented as a 3D array of shape (height, width, 3).
        Each value is a 3D vector in [-1, 1] value range.
    coordinate_system: CartesianCoordinateSystem
        The coordinate system.
    threshold: float(default=0.9)
    """
    value: np.ndarray
    coordinate_system: CartesianCoordinateSystem
    threshold: float = 0.9

    def __post_init__(self):
        if self.value.ndim != 3:
            raise ValueError(f"Normal map must have 3 dimensions, got {self.value.ndim}")
        if self.value.shape[2] != 3:
            raise ValueError(f"Normal map must have 3 channels, got {self.value.shape[2]}")
        if self.value.min() < -1 or self.value.max() > 1:
            raise ValueError(f"Normal map must be in [-1, 1] value range, got {self.value.min()} and {self.value.max()}")

    def __getitem__(
        self, 
        indices: tuple[int, int]
        ) -> NormalValue:
        """
        Get the normal map at the given indices.

        Parameters
        ----------
        indices: tuple[np.ndarray, np.ndarray]
            The indices to get the normal map at.

        Returns
        -------
        NormalValue:
            The normal value at the given indices.
        """
        value = self.value[indices]
        return NormalValue(
            value=value,
            coordinate_system=self.coordinate_system,
            )

    @property
    def horizontal_plane_mask(self) -> np.ndarray:
        """
        Get the horizontal plane mask.
        """
        up_index = self.coordinate_system.up.to_index
        return self.value[:, :, up_index] > self.threshold

    @property
    def vertical_plane_mask(self) -> np.ndarray:
        """
        Get the vertical plane mask.
        """
        up_index = self.coordinate_system.up.to_index
        threshold = 1 - self.threshold
        return np.abs(self.value[:, :, up_index]) < threshold

    @property
    def zero_one_normal_map(self) -> np.ndarray:
        """
        Get the normal map represented as a 3D array of shape (height, width, 3).
        Each value is a 3D vector in [0, 1] value range.
        """
        return (self.value + 1) / 2

    @classmethod
    def from_zero_one_normal_map(
        cls,
        value: np.ndarray,
        coordinate_system: CartesianCoordinateSystem
        ) -> NormalMap:
        """
        Create a normal map from a zero one normal map.

        Parameters
        ----------
        value: np.ndarray
            The normal map represented as a 3D array of shape (height, width, 3).
            Each value is a 3D vector in [0, 1] value range.
        coordinate_system: CartesianCoordinateSystem
            The coordinate system.

        Returns
        -------
        NormalMap:
            The normal map.
        """
        if np.min(value) < 0 or np.max(value) > 1:
            raise ValueError("Zero one normal map must be in [0, 1] value range")
        normal_map = 2 * value - 1 # convert [0, 1] to [-1, 1] value range
        normalized_norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
        normalized_norm[normalized_norm < 1e-6] = 1e-6
        return cls(
            value=normal_map / normalized_norm, 
            coordinate_system=coordinate_system
            )

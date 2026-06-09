"""
Shared NumPy array type aliases for the geometry package.

Dtype is expressed through ``numpy.typing.NDArray``. Shape constraints are
enforced at runtime in dataclass validators and described in class docstrings.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NumericArray: TypeAlias = NDArray[np.number]
FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]
BoolArray: TypeAlias = NDArray[np.bool_]

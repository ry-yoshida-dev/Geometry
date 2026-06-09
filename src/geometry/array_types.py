"""
Shared NumPy array type aliases for the geometry package.

Dtype is expressed through ``numpy.typing.NDArray``. Shape constraints are
enforced at runtime in dataclass validators and described in class docstrings.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating]
BoolArray: TypeAlias = NDArray[np.bool_]

# polar

## Overview

Batched 2D polar coordinates `(radius, angle)` with conversion to and from Cartesian `(u, v)`. `Angle` and `AngleUnit` are re-exported from `units`.

## Components

| Component | Description |
|-----------|-------------|
| [coordinate.py](./coordinate.py) | `PolarCoordinate` with radius shape `(N,)`, Cartesian accessors, and `from_uv` constructor. |

## Examples

```python
import numpy as np
from geometry.planar.polar import PolarCoordinate

polar = PolarCoordinate.from_uv(u=np.array([1.0, 0.0]), v=np.array([0.0, 1.0]))
polar.degree
```

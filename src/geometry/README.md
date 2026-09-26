# geometry

## Overview

Top-level package for 2D (`planar`) and 3D (`spatial`) geometry primitives. The package root re-exports the public 2D API.

## Components

| Component | Description |
|-----------|-------------|
| [planar/](./planar/README.md) | 2D points, lines, vectors, bounding boxes, polar coordinates, and measurements. |
| [spatial/](./spatial/README.md) | 3D points, lines, vectors, normals, and Cartesian coordinate systems. |
| [utils/](./utils/README.md) | Rotation matrix helpers. |
| [array_types.py](./array_types.py) | Shared NumPy type aliases (`NumericScalar`, `NumericArray`, `FloatArray`, `BoolArray`). |

## Examples

```python
import numpy as np
from geometry import Box2DFormat, Boxes2D, Point2D

point = Point2D(np.array([1.0, 2.0]))
boxes = Boxes2D.register(
    value=np.array([[10, 20, 30, 40]]),
    box2d_format=Box2DFormat.XYXY,
)
```

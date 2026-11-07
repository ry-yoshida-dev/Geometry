# Geometry

## Overview

Geometry is a Python package for 2D and 3D geometry primitives: points, lines, vectors, bounding boxes, polar coordinates, surface normals, and rotation helpers.

See [planar](src/geometry/planar/README.md) for 2D and [spatial](src/geometry/spatial/README.md) for 3D components.

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np
from geometry.planar import Point2D, Vector2D
from geometry.spatial import Point3D

point = Point2D(np.array([1.0, 2.0]))
vector = Vector2D(np.array([0.0, 1.0]))
point_3d = Point3D(np.array([1.0, 2.0, 3.0]))
```

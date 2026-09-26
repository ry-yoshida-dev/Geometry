# normal

## Overview

Surface normals in the `[-1, 1]` range tied to a `CartesianCoordinateSystem`, for single pixels or dense normal maps.

## Components

| Component | Description |
|-----------|-------------|
| [normal_value.py](./normal_value.py) | `NormalValue` — a single normal with shape `(3,)`. |
| [normal_map.py](./normal_map.py) | `NormalMap` — shape `(H, W, 3)`; horizontal/vertical plane masks and `[0, 1]` encoding conversion. |

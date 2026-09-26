# box

## Overview

Axis-aligned 2D bounding boxes in image pixel coordinates (x right, y down). Single-box and batched APIs, concrete XYXY/XYWH implementations in per-format subpackages, and shared helpers in `utils/`.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Abstract `Box2D` over `float` or `np.ndarray`. |
| [box.py](./box.py) | Single box — `value` shape `(4,)`. |
| [boxes.py](./boxes.py) | Batch — `value` shape `(N, 4)`. |
| [format.py](./format.py) | `Box2DFormat` enum. |
| [xyxy/](./xyxy/README.md) | Concrete XYXY classes (`Box2D_XYXY`, `Boxes2D_XYXY`). |
| [xywh/](./xywh/README.md) | Concrete XYWH classes (`Box2D_XYWH`, `Boxes2D_XYWH`). |
| [utils/](./utils/README.md) | `BboxCalculator`, `Box2dConverter`, low-level converters. |

## Examples

```python
import numpy as np
from geometry.planar.box import Box2D, Box2DFormat, Boxes2D

boxes = Boxes2D.register(
    value=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
    box2d_format=Box2DFormat.XYXY,
)
box = boxes[0]
subset = boxes[1:]
```

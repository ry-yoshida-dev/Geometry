# xyxy

## Overview

Concrete XYXY bounding boxes: two corners `x1, y1, x2, y2`; requires `x1 < x2` and `y1 < y2`. Prefer `Box2D.register` / `Boxes2D.register` with `Box2DFormat.XYXY` over constructing these classes directly.

## Components

| Component | Description |
|-----------|-------------|
| [box.py](./box.py) | `Box2D_XYXY` — single box, `value` shape `(4,)`. |
| [boxes.py](./boxes.py) | `Boxes2D_XYXY` — batch, `value` shape `(N, 4)`. |

# xywh

## Overview

Concrete XYWH bounding boxes: top-left plus size `x, y, width, height`; requires positive width and height. Prefer `Box2D.register` / `Boxes2D.register` with `Box2DFormat.XYWH` over constructing these classes directly.

## Components

| Component | Description |
|-----------|-------------|
| [box.py](./box.py) | `Box2D_XYWH` — single box, `value` shape `(4,)`. |
| [boxes.py](./boxes.py) | `Boxes2D_XYWH` — batch, `value` shape `(N, 4)`. |

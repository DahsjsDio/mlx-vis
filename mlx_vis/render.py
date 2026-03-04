"""GPU-accelerated frame renderer using MLX Metal."""

import mlx.core as mx
import numpy as np

# cached circle offsets keyed by radius
_OFFSET_CACHE = {}


def _circle_offsets(radius):
    """Pre-compute pixel offsets for a filled circle of given radius."""
    if radius in _OFFSET_CACHE:
        return _OFFSET_CACHE[radius]
    r = int(np.ceil(radius))
    offsets = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dy, dx))
    result = mx.array(np.array(offsets, dtype=np.int32))
    _OFFSET_CACHE[radius] = result
    return result


def render_frame(Y, colors, width, height, xlim, ylim, point_radius=2,
                 bg_color=None):
    """Render scattered points to an RGBA pixel buffer on GPU.

    Returns (height, width, 4) uint8 RGBA array.
    """
    if bg_color is None:
        bg_color = mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)

    n = Y.shape[0]
    xmin, xmax = float(xlim[0]), float(xlim[1])
    ymin, ymax = float(ylim[0]), float(ylim[1])

    # map coords to pixel space
    px = ((Y[:, 0] - xmin) / (xmax - xmin) * (width - 1)).astype(mx.int32)
    py = ((ymax - Y[:, 1]) / (ymax - ymin) * (height - 1)).astype(mx.int32)

    # circle template offsets
    offsets = _circle_offsets(point_radius)  # (k, 2)
    k = offsets.shape[0]

    # expand each point by circle offsets: (n, k)
    all_y = py[:, None] + offsets[None, :, 0]
    all_x = px[:, None] + offsets[None, :, 1]

    # clamp to image bounds
    all_y = mx.clip(all_y, 0, height - 1)
    all_x = mx.clip(all_x, 0, width - 1)

    # flatten to 1D pixel indices
    flat_idx = (all_y * width + all_x).reshape(-1)

    # expand colors: (n, 4) -> (n*k, 4)
    cols_expanded = mx.broadcast_to(colors[:, None, :], (n, k, 4)).reshape(-1, 4)

    # fill background then scatter points (last write wins)
    total_pixels = height * width
    buf = mx.broadcast_to(bg_color[None, :], (total_pixels, 4))
    buf = mx.array(buf)
    buf[flat_idx] = cols_expanded

    return (buf.reshape(height, width, 4) * 255).astype(mx.uint8)

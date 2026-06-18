"""Small image-layout helpers shared across the harness."""

from __future__ import annotations

import numpy as np


def hconcat_uint8(frames: list[np.ndarray]) -> np.ndarray:
    """Stitch uint8 HWC RGB frames left-to-right into a single strip.

    Frames may differ in size (e.g. agentview vs wrist vs depth): each is
    resized — preserving aspect ratio — to the tallest frame's height so they
    line up, then concatenated along the width axis in the given order.
    """
    if not frames:
        raise ValueError("no frames to concatenate")
    if len(frames) == 1:
        return frames[0]

    from PIL import Image

    target_h = max(f.shape[0] for f in frames)
    aligned: list[np.ndarray] = []
    for f in frames:
        h, w = f.shape[:2]
        if h != target_h:
            new_w = max(1, round(w * target_h / h))
            f = np.asarray(
                Image.fromarray(f, "RGB").resize((new_w, target_h), Image.Resampling.BILINEAR), dtype=np.uint8
            )
        aligned.append(f)
    return np.concatenate(aligned, axis=1)

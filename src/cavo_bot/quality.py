from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError


@dataclass(frozen=True, slots=True)
class ImageQuality:
    accepted: bool
    reason: str
    width: int = 0
    height: int = 0
    brightness: float = 0.0
    edge_variance: float = 0.0


def inspect_image(
    image_bytes: bytes,
    *,
    max_bytes: int,
    min_side: int = 224,
    min_brightness: float = 18.0,
    max_brightness: float = 242.0,
    min_edge_variance: float = 8.0,
) -> ImageQuality:
    if not image_bytes:
        return ImageQuality(False, "empty")
    if len(image_bytes) > max_bytes:
        return ImageQuality(False, "too_large")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return ImageQuality(False, "invalid")

    width, height = image.size
    if min(width, height) < min_side:
        return ImageQuality(False, "too_small", width=width, height=height)

    sample = ImageOps.contain(image, (384, 384)).convert("L")
    brightness = float(ImageStat.Stat(sample).mean[0])
    edges = np.asarray(sample.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edge_variance = float(edges.var())
    if brightness < min_brightness:
        reason = "too_dark"
    elif brightness > max_brightness:
        reason = "too_bright"
    elif edge_variance < min_edge_variance:
        reason = "too_blurry"
    else:
        reason = "ok"
    return ImageQuality(
        reason == "ok",
        reason,
        width=width,
        height=height,
        brightness=brightness,
        edge_variance=edge_variance,
    )

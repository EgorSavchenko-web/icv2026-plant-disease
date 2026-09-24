from __future__ import annotations

import io
import zlib

import numpy as np
from PIL import Image
from scipy.ndimage import convolve


MOTION_BLUR_LENGTHS = {1: 5, 2: 11, 3: 19}
LOW_LIGHT_FACTORS = {1: 0.50, 2: 0.30, 3: 0.18}
LOW_LIGHT_READ_NOISE = {1: 0.010, 2: 0.020, 3: 0.035}
JPEG_QUALITIES = {1: 30, 2: 15, 3: 8}
FULL_WELL_PHOTONS = 255.0

CORRUPTIONS = ("motion_blur", "low_light", "jpeg")
SEVERITIES = (1, 2, 3)

CORRUPTION_LABELS = {
    "none": "Clean",
    "motion_blur": "Motion blur",
    "low_light": "Low light + sensor noise",
    "jpeg": "JPEG compression",
}

CORRUPTION_PARAMETERS = {
    "motion_blur": {"kernel_length_px": MOTION_BLUR_LENGTHS},
    "low_light": {"exposure_factor": LOW_LIGHT_FACTORS, "read_noise_sigma": LOW_LIGHT_READ_NOISE},
    "jpeg": {"quality": JPEG_QUALITIES},
}


def seed_from_key(key: str) -> int:
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


def line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float64)
    center = (length - 1) / 2.0
    radians = np.deg2rad(angle_degrees)
    step_y, step_x = -np.sin(radians), np.cos(radians)

    for offset in np.linspace(-center, center, length * 4):
        row = int(round(center + step_y * offset))
        column = int(round(center + step_x * offset))
        kernel[row, column] += 1.0

    return kernel / kernel.sum()


def apply_motion_blur(array: np.ndarray, severity: int, rng) -> np.ndarray:
    kernel = line_kernel(MOTION_BLUR_LENGTHS[severity], float(rng.uniform(0.0, 180.0)))
    blurred = np.empty_like(array)

    for channel in range(array.shape[2]):
        blurred[..., channel] = convolve(array[..., channel], kernel, mode="reflect")

    return blurred


def apply_low_light(array: np.ndarray, severity: int, rng) -> np.ndarray:
    exposure = LOW_LIGHT_FACTORS[severity]
    read_noise = LOW_LIGHT_READ_NOISE[severity]

    photons = np.clip(array * exposure, 0.0, 1.0) * FULL_WELL_PHOTONS
    captured = rng.poisson(photons) / FULL_WELL_PHOTONS
    captured = captured + rng.normal(0.0, read_noise, size=array.shape)

    return np.clip(captured / exposure, 0.0, 1.0)


def apply_jpeg(image: Image.Image, severity: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITIES[severity])
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def corrupt_image(image: Image.Image, corruption: str, severity: int, key: str) -> Image.Image:
    if corruption == "none":
        return image

    if corruption == "jpeg":
        return apply_jpeg(image, severity)

    rng = np.random.default_rng(seed_from_key(f"{key}|{corruption}|{severity}"))
    array = np.asarray(image, dtype=np.float64) / 255.0

    if corruption == "motion_blur":
        array = apply_motion_blur(array, severity, rng)
    elif corruption == "low_light":
        array = apply_low_light(array, severity, rng)
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    return Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8))


def condition_label(corruption: str, severity: int) -> str:
    if corruption == "none":
        return CORRUPTION_LABELS["none"]
    return f"{CORRUPTION_LABELS[corruption]} s{severity}"


def all_conditions():
    return [("none", 0)] + [
        (corruption, severity) for corruption in CORRUPTIONS for severity in SEVERITIES
    ]

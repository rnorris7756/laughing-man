#!/usr/bin/env python3
#
# Chroma-key a solid background to transparency for PNG overlays (HSV + corner key).
#

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import typer
from PIL import Image

app = typer.Typer(
    help=(
        "Replace a chroma-key background with transparency (default: infer key from "
        "corner patches, key in HSV). Use output with: laughing-man --image <out.png>"
    ),
    add_completion=False,
    no_args_is_help=True,
)

_DEFAULT_CORNER_PATCH = 8
_DEFAULT_QUANTIZE = 8
# Normalized HSV distance sqrt(dh²+ds²+dv²) with dh = min circular hue / 90°.
_DEFAULT_HSV_DISTANCE = 0.42


def _parse_key_rgb(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise typer.BadParameter("Expected three comma-separated integers R,G,B.")
    try:
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise typer.BadParameter("R,G,B must be integers.") from e
    for n, name in ((r, "R"), (g, "G"), (b, "B")):
        if not 0 <= n <= 255:
            raise typer.BadParameter(f"{name} must be in 0..255.")
    return (r, g, b)


def infer_key_rgb_from_corner_patches(
    rgb: np.ndarray,
    *,
    patch: int = _DEFAULT_CORNER_PATCH,
    quantize_step: int = _DEFAULT_QUANTIZE,
) -> tuple[int, int, int]:
    """
    Estimate background RGB from the most common quantized color in four corner patches.

    Samples a ``patch``×``patch`` square at each image corner, quantizes RGB to reduce
    noise, then returns the mode (most frequent triple).

    Parameters
    ----------
    rgb
        ``(H, W, 3)`` uint8 RGB image.
    patch
        Edge length of the square sampled at each corner.
    quantize_step
        Bin width per channel before voting (e.g. 8 maps 0–255 to 32 bins).

    Returns
    -------
    tuple[int, int, int]
        Estimated background color ``(R, G, B)``.
    """
    h, w = rgb.shape[:2]
    p = min(patch, h, w)
    if p < 1:
        raise ValueError("Image is too small for corner sampling.")
    corners = (
        rgb[:p, :p],
        rgb[:p, w - p :],
        rgb[h - p :, :p],
        rgb[h - p :, w - p :],
    )
    pix = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    qs = max(1, int(quantize_step))
    q = (pix.astype(np.int32) // qs) * qs
    q = np.clip(q, 0, 255).astype(np.uint8)
    packed = (
        q[:, 0].astype(np.uint32) << 16 | q[:, 1].astype(np.uint32) << 8 | q[:, 2].astype(np.uint32)
    )
    uniq, counts = np.unique(packed, return_counts=True)
    winner = int(uniq[np.argmax(counts)])
    r = (winner >> 16) & 255
    g = (winner >> 8) & 255
    b = winner & 255
    return (int(r), int(g), int(b))


def _hsv_normalized_distance(hsv: np.ndarray, key_hsv: tuple[float, float, float]) -> np.ndarray:
    """
    Per-pixel distance in normalized HSV space (hue uses circular min arc / 90°).

    OpenCV HSV: H ∈ [0, 179] maps to [0°, 360°) in steps of 2°; S, V ∈ [0, 255].
    """
    kh, ks, kv = key_hsv
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    kh_i = int(round(kh))
    d_lin = np.abs(h - kh_i)
    dh = np.minimum(d_lin, 180 - d_lin).astype(np.float32)
    dh_norm = dh / 90.0
    ds_norm = np.abs(s - ks) / 255.0
    dv_norm = np.abs(v - kv) / 255.0
    return np.sqrt(dh_norm * dh_norm + ds_norm * ds_norm + dv_norm * dv_norm)


def chroma_key_hsv(
    rgb: np.ndarray,
    key_rgb: tuple[int, int, int],
    *,
    distance_max: float,
) -> np.ndarray:
    """
    Boolean mask: True where pixel should become transparent.

    Parameters
    ----------
    rgb
        ``(H, W, 3)`` uint8 RGB.
    key_rgb
        Key color in RGB.
    distance_max
        Maximum normalized HSV distance (typical 0.25–0.55).

    Returns
    -------
    numpy.ndarray
        Boolean array shaped ``(H, W)``.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    key_px = np.uint8([[[key_rgb[0], key_rgb[1], key_rgb[2]]]])
    k_bgr = cv2.cvtColor(key_px, cv2.COLOR_RGB2BGR)
    k_hsv = cv2.cvtColor(k_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    kh, ks, kv = float(k_hsv[0]), float(k_hsv[1]), float(k_hsv[2])
    dist = _hsv_normalized_distance(hsv, (kh, ks, kv))
    return dist <= distance_max


def chroma_key_rgb_box(
    rgb: np.ndarray,
    key_rgb: tuple[int, int, int],
    *,
    tolerance: int,
) -> np.ndarray:
    """Boolean mask using per-channel RGB boxes (legacy ``#FF00FF`` style)."""
    kr, kg, kb = key_rgb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dr = np.abs(r.astype(np.int16) - kr)
    dg = np.abs(g.astype(np.int16) - kg)
    db = np.abs(b.astype(np.int16) - kb)
    return (dr <= tolerance) & (dg <= tolerance) & (db <= tolerance)


def chroma_key_rgb_euclidean(
    rgb: np.ndarray,
    key_rgb: tuple[int, int, int],
    *,
    distance_max: float,
) -> np.ndarray:
    """Boolean mask using Euclidean distance in RGB."""
    kr, kg, kb = key_rgb
    r = rgb[..., 0].astype(np.float32) - kr
    g = rgb[..., 1].astype(np.float32) - kg
    b = rgb[..., 2].astype(np.float32) - kb
    dist = np.sqrt(r * r + g * g + b * b)
    return dist <= distance_max


def apply_mask_to_rgba(rgb: np.ndarray, key_mask: np.ndarray) -> Image.Image:
    """Build RGBA image: keyed pixels transparent, RGB zeroed there."""
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    out[..., 3] = np.where(key_mask, 0, 255)
    for c in range(3):
        out[..., c] = np.where(out[..., 3] > 0, out[..., c], 0)
    return Image.fromarray(out, mode="RGBA")


def chroma_key_image(
    image: Image.Image,
    *,
    key_rgb: tuple[int, int, int],
    method: Literal["hsv", "rgb-euclidean", "rgb-box"],
    hsv_distance: float,
    rgb_euclidean_distance: float,
    box_tolerance: int,
) -> Image.Image:
    """
    Apply chroma key and return an RGBA PIL image.

    Parameters
    ----------
    image
        Source image.
    key_rgb
        Background color to remove.
    method
        ``hsv`` (default behavior), ``rgb-euclidean``, or ``rgb-box``.
    hsv_distance
        Normalized HSV distance threshold for ``method='hsv'``.
    rgb_euclidean_distance
        RGB Euclidean threshold for ``method='rgb-euclidean'``.
    box_tolerance
        Per-channel tolerance for ``method='rgb-box'``.

    Returns
    -------
    PIL.Image.Image
        RGBA result.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if method == "hsv":
        mask = chroma_key_hsv(rgb, key_rgb, distance_max=hsv_distance)
    elif method == "rgb-euclidean":
        mask = chroma_key_rgb_euclidean(rgb, key_rgb, distance_max=rgb_euclidean_distance)
    else:
        mask = chroma_key_rgb_box(rgb, key_rgb, tolerance=box_tolerance)
    return apply_mask_to_rgba(rgb, mask)


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Source image (e.g. chroma-backed PNG from an image model).",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        writable=True,
        help="Output PNG path. Default: <input stem>-keyed.png next to the input.",
    ),
    key: str | None = typer.Option(
        None,
        "--key",
        "-k",
        help="Override auto key as R,G,B (e.g. 255,0,255). Default: infer from corners.",
    ),
    corner_patch: int = typer.Option(
        _DEFAULT_CORNER_PATCH,
        "--corner-patch",
        min=1,
        help="Square size (px) sampled at each of the four corners for key inference.",
    ),
    quantize: int = typer.Option(
        _DEFAULT_QUANTIZE,
        "--quantize",
        min=1,
        max=64,
        help="Quantization step for corner color voting (reduces noise before mode).",
    ),
    distance: float = typer.Option(
        _DEFAULT_HSV_DISTANCE,
        "--distance",
        "-d",
        min=0.0,
        help="Key threshold: for --method hsv, normalized HSV distance (typical 0.3–0.55).",
    ),
    rgb_distance: float = typer.Option(
        45.0,
        "--rgb-distance",
        min=0.0,
        help="Max RGB Euclidean distance when using --method rgb-euclidean.",
    ),
    method: str = typer.Option(
        "hsv",
        "--method",
        "-m",
        help="hsv (default) | rgb-euclidean | rgb-box",
    ),
    tolerance: int = typer.Option(
        40,
        "--tolerance",
        "-t",
        min=0,
        max=255,
        help="Per-channel tolerance for --method rgb-box.",
    ),
    red: int = typer.Option(255, "--red", min=0, max=255, help="Key R for rgb-box (with --method rgb-box)."),
    green: int = typer.Option(0, "--green", min=0, max=255, help="Key G for rgb-box."),
    blue: int = typer.Option(255, "--blue", min=0, max=255, help="Key B for rgb-box."),
) -> None:
    """Write a PNG with the background color replaced by transparent alpha."""
    m = method.lower().strip()
    if m not in ("hsv", "rgb-euclidean", "rgb-box"):
        raise typer.BadParameter("--method must be hsv, rgb-euclidean, or rgb-box.")

    out = output_path
    if out is None:
        out = input_path.parent / f"{input_path.stem}-keyed.png"

    img = Image.open(input_path)
    rgb_arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    if m == "rgb-box":
        key_rgb = (red, green, blue)
        key_label = "rgb-box"
    elif key is not None:
        key_rgb = _parse_key_rgb(key)
        key_label = "manual --key"
    else:
        key_rgb = infer_key_rgb_from_corner_patches(
            rgb_arr,
            patch=corner_patch,
            quantize_step=quantize,
        )
        key_label = "corner mode"

    typer.echo(
        f"Key RGB ({key_label}): {key_rgb[0]} {key_rgb[1]} {key_rgb[2]}",
        err=True,
    )

    rgba = chroma_key_image(
        img,
        key_rgb=key_rgb,
        method=m,  # type: ignore[arg-type]
        hsv_distance=distance,
        rgb_euclidean_distance=rgb_distance,
        box_tolerance=tolerance,
    )

    rgba.save(out, format="PNG", compress_level=6)
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()

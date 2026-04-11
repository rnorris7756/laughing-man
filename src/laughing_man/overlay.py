"""PIL-based Laughing Man overlay assets and rotation cache helpers."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import typer
from loguru import logger
from PIL import Image, ImageChops, ImageDraw

from laughing_man.constants import ROT_IMAGE_NAME, STABLE_IMAGE_NAME, TMP_OFFSET


def build_rotated_overlay_frame(
    *,
    rot_angle: int,
    st_img: Image.Image,
    st_alpha: Image.Image,
    rot_img: Image.Image,
    rot_alpha: Image.Image,
    im_sz: tuple[int, int],
    tmp_offset: float,
    min_dim: int,
    custom_static_overlay: bool = False,
) -> Image.Image:
    """
    Composite stable art, rotating text, and masks; resize to the square ROI size.

    When ``custom_static_overlay`` is True (``--image`` with a user PNG), the
    stock white-ellipse backdrop and rotating layer are skipped: the overlay is
    only ``st_img`` composited over black using ``st_alpha``, so transparent
    pixels stay dark and do not pick up a white disk behind the art.

    Parameters
    ----------
    rot_angle
        Rotation in degrees (typically a multiple of ``ROT_RESO``). Ignored when
        ``custom_static_overlay`` is True.
    st_img, st_alpha, rot_img, rot_alpha
        Source layers (``st_*`` static, ``rot_*`` rotated each frame).
    im_sz
        ``(width, height)`` of ``st_img``.
    tmp_offset
        Ellipse inset factor (same as ``TMP_OFFSET``).
    min_dim
        Edge length of the square overlay after resize.
    custom_static_overlay
        If True, build a single-layer composite for user-supplied overlay art.

    Returns
    -------
    Image.Image
        RGB ``Image`` ready for ``numpy.asarray(..., dtype=uint8)`` in the loop.
    """
    if custom_static_overlay:
        backing = Image.new("RGB", st_img.size, (0, 0, 0))
        tmp_img = Image.composite(st_img, backing, st_alpha)
        return tmp_img.resize((min_dim, min_dim))

    comb_img = Image.new("RGB", st_img.size)
    draw_img = ImageDraw.Draw(comb_img)
    draw_img.ellipse(
        (
            im_sz[0] * tmp_offset,
            im_sz[1] * tmp_offset,
            im_sz[0] * (1 - tmp_offset),
            im_sz[1] * (1 - tmp_offset),
        ),
        fill="white",
    )
    rot_nearest = rot_img.rotate(rot_angle, Image.Resampling.NEAREST)
    rot_a_nearest = rot_alpha.rotate(rot_angle, Image.Resampling.NEAREST)
    tmp_img = Image.composite(
        st_img,
        Image.composite(rot_nearest, comb_img, rot_a_nearest),
        st_alpha,
    )
    return tmp_img.resize((min_dim, min_dim))


def load_overlay_images(
    overlay_image: Path | None,
) -> tuple[Image.Image, Image.Image]:
    """
    Load stable and rotating overlay layers.

    When ``overlay_image`` is set, that file is the stable layer (RGBA); the
    rotating layer is a same-size fully transparent image so the overlay does
    not spin arbitrary artwork.

    Parameters
    ----------
    overlay_image
        Path to a PNG/JPEG/etc. to use instead of the package default assets
        (``laughing_man.assets``: ``limg.png`` / ``ltext.png``). If None, those
        bundled files are used.

    Returns
    -------
    tuple[Image.Image, Image.Image]
        ``(st_img, rot_img)`` as RGBA :class:`PIL.Image.Image` instances.
    """
    if overlay_image is not None:
        p = overlay_image.expanduser().resolve()
        if not p.is_file():
            logger.error("Overlay image not found: {}", p)
            raise typer.Exit(code=1)
        try:
            st_img = Image.open(p).convert("RGBA")
        except OSError as e:
            logger.error("Could not read overlay image {}: {}", p, e)
            raise typer.Exit(code=1) from e
        st_img.load()
        rot_img = Image.new("RGBA", st_img.size, (0, 0, 0, 0))
        return st_img, rot_img

    root = files("laughing_man.assets")
    st_img = _load_bundled_rgba(root.joinpath(STABLE_IMAGE_NAME))
    rot_img = _load_bundled_rgba(root.joinpath(ROT_IMAGE_NAME))
    return st_img, rot_img


def _load_bundled_rgba(path: Traversable) -> Image.Image:
    """Load a PNG from package resources as RGBA."""
    with path.open("rb") as f:
        img = Image.open(f).convert("RGBA")
    img.load()
    return img

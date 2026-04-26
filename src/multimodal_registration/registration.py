"""
Image registration utilities.

Provides phase cross-correlation–based translation estimation between a DCT
volume (or any 3-D array) and a PCT reference volume.  All heavy computation
is dispatched to the active backend (CUDA via CuPy when available, CPU
otherwise).
"""
from __future__ import annotations

import numpy as np

from . import backends
from .dct import DCT
from .pct import ReferencePCT


def find_shift(
    fixed: np.ndarray,
    moving: np.ndarray,
    upsample_factor: int = 10,
) -> np.ndarray:
    """Estimate the translation that aligns *moving* to *fixed*.

    Uses phase cross-correlation (Fourier-domain normalised
    cross-power spectrum), which is robust to intensity differences
    between modalities.

    Parameters
    ----------
    fixed:
        Reference array (e.g. PCT volume or a derived binary mask).
    moving:
        Array to register (e.g. DCT mask after upscaling and padding).
    upsample_factor:
        Sub-pixel precision: a value of 10 gives 0.1-voxel accuracy.

    Returns
    -------
    shift : ndarray, shape (ndim,)
        Translation in voxels that must be applied to *moving* to align it
        with *fixed* (positive = shift right/down).
    """
    try:
        if backends.backend_name() == "cuda":
            from cucim.skimage.registration import phase_cross_correlation
        else:
            from skimage.registration import phase_cross_correlation
    except ImportError:
        from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(
        fixed.astype(float),
        moving.astype(float),
        upsample_factor=upsample_factor,
    )
    return shift


def register(
    dct: DCT,
    pct: ReferencePCT,
    dct_field: str = "Mask",
    pct_field: str = "mask",
    upsample_factor: int = 10,
) -> np.ndarray:
    """High-level registration: find the translation between a DCT field and
    the PCT volume, apply it to **all** DCT volumes, and return the shift.

    Prerequisites
    -------------
    Call ``dct.upscale(pct.voxel_size)`` and ``dct.pad(pct.shape)`` before
    calling this function so that the DCT and PCT arrays share the same
    shape and voxel size.

    Parameters
    ----------
    dct:
        DCT object (already upscaled and padded to PCT frame).
    pct:
        PCT reference object.
    dct_field:
        Attribute name on *dct* used as the moving image (default ``'Mask'``).
    pct_field:
        Attribute name on *pct* used as the fixed image.  Defaults to
        ``pct.mask`` (Otsu-thresholded binary mask).
    upsample_factor:
        Sub-pixel accuracy passed to :func:`find_shift`.

    Returns
    -------
    shift : ndarray, shape (3,)
        Applied translation in voxels.
    """
    fixed = getattr(pct, pct_field)
    moving = getattr(dct, dct_field)

    shift = find_shift(fixed, moving, upsample_factor=upsample_factor)
    dct.shift(tuple(shift.tolist()))
    return shift


def overlay_check(
    dct: DCT,
    pct: ReferencePCT,
    dct_field: str = "Mask",
    plane: str = "xz",
    slice_n: int | None = None,
    alpha: float = 0.5,
):
    """Quick visual check: overlay a DCT field with the PCT volume.

    Parameters
    ----------
    dct:
        DCT object (upscaled, padded, and shifted to PCT frame).
    pct:
        PCT reference object.
    dct_field:
        Attribute name on *dct* to overlay (default ``'Mask'``).
    plane:
        Slice plane (``'xy'``, ``'yz'``, or ``'xz'``).
    slice_n:
        Slice index. Defaults to mid-point.
    alpha:
        Opacity of the DCT overlay (0–1).
    """
    import matplotlib.pyplot as plt

    plane = plane.lower()
    axis_map = {
        "xy": (2, lambda arr, s: arr[:, :, s]),
        "yx": (2, lambda arr, s: arr[:, :, s]),
        "yz": (1, lambda arr, s: arr[:, s, :]),
        "zy": (1, lambda arr, s: arr[:, s, :]),
        "xz": (0, lambda arr, s: arr[s, :, :]),
        "zx": (0, lambda arr, s: arr[s, :, :]),
    }
    if plane not in axis_map:
        raise ValueError(f"Unknown plane '{plane}'. Choose from xy, yz, xz.")

    axis, slicer = axis_map[plane]
    ref_vol = pct.vol
    dct_vol = getattr(dct, dct_field)

    if slice_n is None:
        slice_n = ref_vol.shape[axis] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    pct_slice = slicer(ref_vol, slice_n)
    dct_slice = slicer(dct_vol, slice_n)

    axes[0].imshow(pct_slice, cmap="gray")
    axes[0].set_title("PCT (fixed)")

    axes[1].imshow(dct_slice, cmap="hot")
    axes[1].set_title(f"DCT {dct_field} (moving)")

    axes[2].imshow(pct_slice, cmap="gray")
    axes[2].imshow(dct_slice, cmap="hot", alpha=alpha)
    axes[2].set_title("Overlay")

    plt.tight_layout()
    plt.show()

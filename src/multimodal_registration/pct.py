from __future__ import annotations

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from skimage.filters import threshold_otsu

from . import backends
from .utils import get_extension


class ReferencePCT:
    """Phase Contrast Tomography reference volume.

    Supports TIFF/TIF and HDF5 file formats.

    Parameters
    ----------
    pct_ref_path:
        Path to the PCT volume file.
    h5_key:
        Dataset key inside the HDF5 file. Required when the file is HDF5.
    closing_radius:
        If given, apply a spherical binary closing with this radius (in
        voxels) to the Otsu mask. Useful for filling internal holes before
        registration. GPU-accelerated when CUDA is available.
    """

    def __init__(
        self,
        pct_ref_path: str,
        h5_key: str | None = None,
        closing_radius: int | None = None,
    ):
        self.path = pct_ref_path
        self.im_format = get_extension(self.path)
        self.h5_key = h5_key

        if self.im_format in ("h5", "hdf5") and self.h5_key is None:
            raise ValueError(
                "Please provide an h5_key to load your volume from an HDF5 file."
            )

        self.vol: np.ndarray = self._load_volume()
        self.shape: tuple[int, ...] = self.vol.shape
        self.mask: np.ndarray = self._compute_mask(closing_radius)

    def _compute_mask(self, closing_radius: int | None) -> np.ndarray:
        mask = self.vol > threshold_otsu(self.vol)
        if closing_radius is None:
            return mask

        nd = backends.ndimage()
        r = closing_radius
        # Build a spherical structuring element
        grid = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        struct = (grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) <= r**2

        d_mask = backends.to_device(mask)
        d_struct = backends.to_device(struct)
        closed = nd.binary_closing(d_mask, structure=d_struct)
        return backends.to_numpy(closed)

    def _load_volume(self) -> np.ndarray:
        if self.im_format in ("tiff", "tif"):
            try:
                return sk.io.imread(self.path)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"PCT file not found: {self.path}") from e

        if self.im_format in ("h5", "hdf5"):
            try:
                with h5py.File(self.path, "r") as hin:
                    return hin[self.h5_key][:]
            except FileNotFoundError as e:
                raise FileNotFoundError(f"PCT file not found: {self.path}") from e

        raise ValueError(
            f"Unsupported format '{self.im_format}'. Expected tiff, tif, h5, or hdf5."
        )

    def plot(self, plane: str = "xz", slice_n: int | None = None):
        """Plot a 2-D slice of the PCT volume.

        Parameters
        ----------
        plane:
            One of ``'xy'``, ``'yz'``, ``'xz'`` (and their reverses).
        slice_n:
            Index along the normal axis. Defaults to the mid-point.
        """
        plane = plane.lower()
        axis_map = {
            "xy": (2, lambda s: self.vol[:, :, s]),
            "yx": (2, lambda s: self.vol[:, :, s]),
            "yz": (1, lambda s: self.vol[:, s, :]),
            "zy": (1, lambda s: self.vol[:, s, :]),
            "xz": (0, lambda s: self.vol[s, :, :]),
            "zx": (0, lambda s: self.vol[s, :, :]),
        }
        if plane not in axis_map:
            raise ValueError(f"Unknown plane '{plane}'. Choose from xy, yz, xz.")

        axis, slicer = axis_map[plane]
        if slice_n is None:
            slice_n = self.shape[axis] // 2

        plt.figure()
        plt.imshow(slicer(slice_n), cmap="gray")
        plt.title(f"PCT  plane={plane}  slice={slice_n}")
        plt.show()


# Keep old name as an alias for backward compatibility
Reference_PCT = ReferencePCT

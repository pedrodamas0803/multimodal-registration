from __future__ import annotations

import numpy as np
import h5py
import matplotlib.pyplot as plt

from . import backends


class DCT:
    """Diffraction Contrast Tomography volume.

    Loads all datasets stored under the ``DS`` group of an HDF5 file and
    exposes them as instance attributes (e.g. ``self.GIDvol``, ``self.IPF001``,
    ``self.Mask``).  Volumetric operations (upscale, pad, shift) automatically
    use the active compute backend (CUDA if available, otherwise CPU).

    Parameters
    ----------
    dct_ds_path:
        Path to the DCT HDF5 file.
    """

    def __init__(self, dct_ds_path: str):
        self._vol_keys: list[str] = []

        with h5py.File(dct_ds_path, "r") as hin:
            for key in hin["DS"].keys():
                data = hin[f"DS/{key}"][:]
                setattr(self, key, data)
                if data.ndim >= 3:
                    self._vol_keys.append(key)

        self.shape: tuple[int, ...] = self.GIDvol.shape
        self.voxel_size: float = float(self.VoxSize[0, 0])

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_ipf(self, plane: str = "xz", slice_n: int | None = None):
        """Plot a 2-D slice of the IPF001 volume.

        Parameters
        ----------
        plane:
            One of ``'xy'``, ``'yz'``, ``'xz'`` (and their reverses).
        slice_n:
            Index along the normal axis. Defaults to the mid-point.
        """
        plane = plane.lower()
        axis_map = {
            "xy": (2, lambda s: self.IPF001[:, :, s]),
            "yx": (2, lambda s: self.IPF001[:, :, s]),
            "yz": (1, lambda s: self.IPF001[:, s, :]),
            "zy": (1, lambda s: self.IPF001[:, s, :]),
            "xz": (0, lambda s: self.IPF001[s, :, :]),
            "zx": (0, lambda s: self.IPF001[s, :, :]),
        }
        if plane not in axis_map:
            raise ValueError(f"Unknown plane '{plane}'. Choose from xy, yz, xz.")

        axis, slicer = axis_map[plane]
        if slice_n is None:
            slice_n = self.shape[axis] // 2

        plt.figure()
        plt.imshow(slicer(slice_n))
        plt.title(f"IPF001  plane={plane}  slice={slice_n}")
        plt.show()

    # ------------------------------------------------------------------
    # Volumetric operations
    # ------------------------------------------------------------------

    def upscale(self, target_vox_size: float) -> None:
        """Upscale all 3-D volumes in-place to match *target_vox_size*.

        The zoom factor is ``self.voxel_size / target_vox_size``.
        Integer arrays (grain IDs, masks) use nearest-neighbour
        interpolation (``order=0``); float/RGB arrays use linear
        interpolation (``order=1``).

        Parameters
        ----------
        target_vox_size:
            Voxel size of the reference volume (e.g. PCT voxel size in µm).
        """
        nd = backends.ndimage()
        factor = self.voxel_size / target_vox_size

        for key in self._vol_keys:
            arr: np.ndarray = getattr(self, key)
            order = 0 if np.issubdtype(arr.dtype, np.integer) else 1

            if arr.ndim == 3:
                zoom_factors = [factor, factor, factor]
            elif arr.ndim == 4:
                # Last axis is a channel (e.g. RGB colour) — do not zoom it.
                zoom_factors = [factor, factor, factor, 1.0]
            else:
                continue

            d_arr = backends.to_device(arr.astype(float) if order > 0 else arr)
            zoomed = nd.zoom(d_arr, zoom_factors, order=order)
            setattr(self, key, backends.to_numpy(zoomed))

        self.voxel_size = target_vox_size
        self.shape = self.GIDvol.shape

    def pad(self, target_shape: tuple[int, int, int]) -> None:
        """Centre-pad all 3-D volumes in-place to *target_shape*.

        Parameters
        ----------
        target_shape:
            Desired ``(nx, ny, nz)`` shape, typically the shape of the
            reference PCT volume.
        """
        pad_width = self._calculate_pad_width(target_shape)

        for key in self._vol_keys:
            arr: np.ndarray = getattr(self, key)
            if arr.ndim == 3:
                pw = pad_width
            elif arr.ndim == 4:
                pw = pad_width + ((0, 0),)  # leave channel axis alone
            else:
                continue
            setattr(self, key, np.pad(arr, pw))

        self.shape = self.GIDvol.shape

    def flip(self, axis: int = 0) -> None:
        """Flip all 3-D volumes in-place along *axis*.

        Parameters
        ----------
        axis:
            Spatial axis to flip (0, 1, or 2). Default is 0 (vertical).
        """
        for key in self._vol_keys:
            arr: np.ndarray = getattr(self, key)
            setattr(self, key, np.flip(arr, axis=axis).copy())

    def shift(self, shifts: tuple[float, float, float]) -> None:
        """Shift all 3-D volumes in-place by *shifts* voxels.

        Integer arrays are shifted with nearest-neighbour interpolation
        (``order=0``); float arrays use linear interpolation (``order=1``).

        Parameters
        ----------
        shifts:
            ``(dx, dy, dz)`` shift in voxels along each axis.
        """
        nd = backends.ndimage()

        for key in self._vol_keys:
            arr: np.ndarray = getattr(self, key)
            order = 0 if np.issubdtype(arr.dtype, np.integer) else 1

            if arr.ndim == 3:
                sv = list(shifts)
            elif arr.ndim == 4:
                sv = list(shifts) + [0.0]
            else:
                continue

            d_arr = backends.to_device(arr.astype(float) if order > 0 else arr)
            shifted = nd.shift(d_arr, sv, order=order)
            setattr(self, key, backends.to_numpy(shifted))

        self.shape = self.GIDvol.shape

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_pad_width(
        self, target_shape: tuple[int, int, int]
    ) -> tuple[tuple[int, int], ...]:
        """Return ``((before, after), ...)`` padding for each spatial axis.

        Padding is symmetric; any extra pixel goes to the *after* side.
        """
        pad_width = []
        for current, target in zip(self.shape, target_shape):
            diff = target - current
            before = diff // 2
            after = diff - before
            pad_width.append((before, after))
        return tuple(pad_width)

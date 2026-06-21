from __future__ import annotations

import gc
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self._attr_keys: list[str] = []
        self._tmpdir = tempfile.TemporaryDirectory()
        self._workdir: str = self._tmpdir.name

        with h5py.File(dct_ds_path, "r") as hin:
            for key in hin["DS"].keys():
                data = hin[f"DS/{key}"][:]
                if data.ndim >= 3:
                    setattr(self, key, self._to_memmap(key, data))
                    del data
                    self._vol_keys.append(key)
                else:
                    setattr(self, key, data)
                    self._attr_keys.append(key)

        self.shape: tuple[int, ...] = self.GIDvol.shape
        self.voxel_size: float = float(self.VoxSize[0, 0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the current state to *directory*.

        All volume fields are written as raw binary files and a ``meta.json``
        records shapes, dtypes, and scalar attributes.  Restore with
        :meth:`from_cache`.

        Parameters
        ----------
        directory:
            Destination directory (created if it does not exist).
        """
        os.makedirs(directory, exist_ok=True)

        meta: dict = {
            "voxel_size": self.voxel_size,
            "shape": list(self.shape),
            "vol_keys": self._vol_keys,
            "attr_keys": self._attr_keys,
            "fields": {},
        }

        for key in self._vol_keys:
            arr = getattr(self, key)
            meta["fields"][key] = {"dtype": arr.dtype.str, "shape": list(arr.shape)}
            dst = os.path.join(directory, f"{key}.dat")
            mm = np.memmap(dst, dtype=arr.dtype, mode="w+", shape=arr.shape)
            mm[:] = arr
            mm.flush()

        attrs = {k: getattr(self, k) for k in self._attr_keys}
        if attrs:
            np.savez(os.path.join(directory, "attrs.npz"), **attrs)

        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def from_cache(cls, directory: str) -> DCT:
        """Restore a :class:`DCT` from a directory written by :meth:`save`.

        The saved directory becomes the working directory, so subsequent
        operations (pad, shift) update the files in-place.

        Parameters
        ----------
        directory:
            Directory previously written by :meth:`save`.
        """
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        obj: DCT = object.__new__(cls)
        obj._tmpdir = None          # directory is user-managed
        obj._workdir = directory
        obj._vol_keys = meta["vol_keys"]
        obj._attr_keys = meta.get("attr_keys", [])
        obj.voxel_size = meta["voxel_size"]
        obj.shape = tuple(meta["shape"])

        for key in obj._vol_keys:
            info = meta["fields"][key]
            dtype = np.dtype(info["dtype"])
            shape = tuple(info["shape"])
            path = os.path.join(directory, f"{key}.dat")
            setattr(obj, key, np.memmap(path, dtype=dtype, mode="r+", shape=shape))

        attrs_path = os.path.join(directory, "attrs.npz")
        if os.path.exists(attrs_path):
            saved = np.load(attrs_path, allow_pickle=True)
            for k in obj._attr_keys:
                setattr(obj, k, saved[k])

        return obj

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

    def upscale(
        self,
        factor: float,
        keys: list[str] | None = None,
    ) -> None:
        """Upscale 3-D volumes in-place by *factor*.

        Integer arrays (grain IDs, masks) use nearest-neighbour interpolation
        (``order=0``); float/RGB arrays use linear interpolation (``order=1``).

        Parameters
        ----------
        factor:
            Zoom factor to apply along each spatial axis (e.g. 2.0 doubles
            the resolution). Typically ``dct.voxel_size / pct.voxel_size``.
        keys:
            Subset of field names to upscale. ``None`` (default) processes
            all fields. Useful when some fields can be recomputed more cheaply
            than zooming them.
        """
        nd = backends.ndimage()
        keys_to_zoom = self._vol_keys if keys is None else [k for k in keys if k in self._vol_keys]

        def _zoom_3d(arr_3d: np.ndarray, order: int) -> np.ndarray:
            d = backends.to_device(arr_3d.astype(float) if order > 0 else arr_3d)
            return backends.to_numpy(nd.zoom(d, [factor, factor, factor], order=order))

        if backends.backend_name() == "cuda":
            # GPU already parallelises within each kernel; keep sequential to
            # avoid multi-thread CuPy stream conflicts.
            for key in keys_to_zoom:
                arr: np.ndarray = getattr(self, key)
                order = 0 if np.issubdtype(arr.dtype, np.integer) else 1
                if arr.ndim == 3:
                    zoomed = _zoom_3d(arr, order)
                elif arr.ndim == 4:
                    zoomed = np.stack(
                        [_zoom_3d(arr[..., c], order) for c in range(arr.shape[-1])],
                        axis=-1,
                    )
                else:
                    continue
                setattr(self, key, self._to_memmap(key, zoomed))
                del zoomed
                gc.collect()
        else:
            # Flatten all work to 3-D tasks — 3-D fields and individual
            # channels of 4-D fields all compete for the same workers.
            # scipy.ndimage.zoom releases the GIL so threads run in parallel.
            tasks: list[tuple[str, int | None]] = []
            for key in keys_to_zoom:
                arr = getattr(self, key)
                if arr.ndim == 3:
                    tasks.append((key, None))
                elif arr.ndim == 4:
                    for c in range(arr.shape[-1]):
                        tasks.append((key, c))

            n_workers = min(len(tasks), os.cpu_count() or 1)
            channel_buf: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_task = {
                    pool.submit(
                        _zoom_3d,
                        getattr(self, key) if ch is None else getattr(self, key)[..., ch],
                        0 if np.issubdtype(getattr(self, key).dtype, np.integer) else 1,
                    ): (key, ch)
                    for key, ch in tasks
                }

                for future in as_completed(future_to_task):
                    key, ch = future_to_task[future]
                    zoomed = future.result()
                    if ch is None:
                        setattr(self, key, self._to_memmap(key, zoomed))
                        del zoomed
                        gc.collect()
                    else:
                        channel_buf[key][ch] = zoomed
                        n_ch = getattr(self, key).shape[-1]
                        if len(channel_buf[key]) == n_ch:
                            stacked = np.stack(
                                [channel_buf[key][c] for c in range(n_ch)], axis=-1
                            )
                            setattr(self, key, self._to_memmap(key, stacked))
                            del stacked, channel_buf[key]
                            gc.collect()

        self.voxel_size /= factor
        self.shape = self.GIDvol.shape

    def pad(self, target_shape: tuple[int, int, int], keys: list[str] | None = None) -> None:
        """Centre-pad 3-D volumes in-place to *target_shape*.

        Parameters
        ----------
        target_shape:
            Desired ``(nx, ny, nz)`` shape, typically the shape of the
            reference PCT volume.
        keys:
            Subset of field names to pad. ``None`` (default) processes all
            fields.
        """
        pad_width = self._calculate_pad_width(target_shape)
        keys_to_pad = self._vol_keys if keys is None else [k for k in keys if k in self._vol_keys]

        for key in keys_to_pad:
            arr: np.ndarray = getattr(self, key)
            if arr.ndim == 3:
                pw = pad_width
            elif arr.ndim == 4:
                pw = pad_width + ((0, 0),)  # leave channel axis alone
            else:
                continue

            padded_shape = tuple(s + b + a for s, (b, a) in zip(arr.shape, pw))

            # Copy source data before creating the new memmap: the padded
            # memmap uses the same backing file as arr (mode="w+" truncates it),
            # so arr must be read into memory first or the source becomes zeros.
            data = np.array(arr)
            del arr

            path = os.path.join(self._workdir, f"{key}.dat")
            mm = np.memmap(path, dtype=data.dtype, mode="w+", shape=padded_shape)
            slices = tuple(slice(b, b + s) for (b, _), s in zip(pw, data.shape))
            mm[slices] = data
            mm.flush()

            del data
            gc.collect()
            setattr(self, key, mm)

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
            shifted = backends.to_numpy(nd.shift(d_arr, sv, order=order))
            del d_arr
            setattr(self, key, self._to_memmap(key, shifted))
            del shifted
            gc.collect()

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

    def _to_memmap(self, key: str, arr: np.ndarray) -> np.memmap:
        """Write *arr* to a per-key memory-mapped file and return the memmap."""
        path = os.path.join(self._workdir, f"{key}.dat")
        mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        return mm

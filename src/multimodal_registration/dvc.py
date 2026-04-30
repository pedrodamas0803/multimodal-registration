"""
DVC (Digital Volume Correlation) results loader.

Supports two file formats and both single-step and multi-step datasets:

**VTK** (``.vtk``, ``.vts``, ``.vtu``)
    Pass a single path for one-step data, or a **list of paths** (one per
    loading step) for multi-step data.  Each file must contain the same
    displacement and strain field names.

**MATLAB v7.3 / HDF5** (``.mat``, ``.h5``, ``.hdf5``)
    Three layouts are recognised automatically:

    * **Single step** — displacement dataset at ``displacement_key``.
    * **Multi-group** — one group per step; supply ``step_keys`` as a list
      of group names (e.g. ``["step_1", "step_2"]``); the displacement is
      read from ``<group>/<displacement_key>``.
    * **5-D array** — displacement dataset with shape
      ``(n_steps, nx, ny, nz, 3)``; steps are indexed along axis 0.

Use :meth:`DVC.select_step` to switch between loading steps.
``dvc.displacement`` always reflects the currently selected step.

Install pyvista to load VTK files::

    pip install pyvista
"""
from __future__ import annotations

import warnings

import numpy as np

from .utils import get_extension


class DVC:
    """DVC result volume, with optional multi-step support.

    Parameters
    ----------
    path:
        Path (or list of paths for multi-step VTK) to the DVC result file.
    displacement_key:
        Field / dataset name for the displacement array (default
        ``'displacement'`` for VTK, ``'U'`` for HDF5).
    strain_keys:
        Field names to load as strain data (*VTK only*).
    step_keys:
        *HDF5/MAT multi-group only.* List of HDF5 group names, one per
        loading step.  The displacement is read from
        ``<step_key>/<displacement_key>``.
    step:
        Initial active step index (0-based).  Negative values count from
        the end (default ``-1`` → last step).
    origin_key:
        *HDF5/MAT only.* Dataset key for the ROI origin array.
    spacing_key:
        *HDF5/MAT only.* Dataset key for the grid spacing array.
    origin:
        Explicit ROI origin ``(x0, y0, z0)`` in PCT voxel coordinates.
        Overrides *origin_key*.
    spacing:
        Explicit grid spacing ``(dx, dy, dz)`` in voxels.
        Overrides *spacing_key*.
    """

    _VTK_EXTENSIONS = {"vtk", "vts", "vtu", "vtr", "vtm"}
    _H5_EXTENSIONS  = {"mat", "h5", "hdf5", "res§"}

    def __init__(
        self,
        path: str | list[str],
        displacement_key: str = "displacement",
        strain_keys: list[str] | None = None,
        step_keys: list[str] | None = None,
        step: int = -1,
        origin_key: str | None = "origin",
        spacing_key: str | None = "spacing",
        origin: tuple[float, float, float] | None = None,
        spacing: tuple[float, float, float] | None = None,
    ):
        # --- store loader configuration ------------------------------------
        self._displacement_key = displacement_key
        self._strain_keys      = strain_keys
        self._origin_key       = origin_key
        self._spacing_key      = spacing_key
        self._origin_override  = origin
        self._spacing_override = spacing

        self.strain: dict[str, np.ndarray] = {}
        self.origin:  tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

        # --- detect mode ---------------------------------------------------
        if isinstance(path, list):
            self._mode   = "vtk_multi"
            self._paths  = path
            self.path    = path[0]
            self.fmt     = get_extension(self.path)
            self.n_steps = len(path)
            # Read geometry once from the first file
            self._read_vtk_geometry(self.path)

        else:
            self.path = path
            self.fmt  = get_extension(path)

            if self.fmt in self._VTK_EXTENSIONS:
                self._mode   = "vtk_single"
                self.n_steps = 1

            elif self.fmt in self._H5_EXTENSIONS:
                if step_keys is not None:
                    self._mode      = "h5_group"
                    self._step_keys = step_keys
                    self.n_steps    = len(step_keys)
                else:
                    # Peek at the array shape to detect 5-D layout
                    self._mode, self.n_steps = self._probe_h5(
                        path, displacement_key)
            else:
                raise ValueError(
                    f"Unsupported format '{self.fmt}'. "
                    f"Expected one of: "
                    f"{self._VTK_EXTENSIONS | self._H5_EXTENSIONS}."
                )

        # --- resolve initial step index ------------------------------------
        self._step = step if step >= 0 else self.n_steps + step
        if not 0 <= self._step < self.n_steps:
            raise IndexError(
                f"step={step} out of range for n_steps={self.n_steps}."
            )

        # --- load active step ----------------------------------------------
        self._load_step(self._step)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Currently active step index (0-based)."""
        return self._step

    def select_step(self, step: int) -> None:
        """Switch to a different loading step and reload the displacement.

        Parameters
        ----------
        step:
            Step index (0-based).  Negative values count from the end.
        """
        if step < 0:
            step = self.n_steps + step
        if not 0 <= step < self.n_steps:
            raise IndexError(
                f"step={step} out of range for n_steps={self.n_steps}."
            )
        self._step = step
        self._load_step(step)

    def available_fields(self) -> list[str]:
        """Return all point- and cell-data field names in a VTK file.

        Useful for discovering field names before loading.  Returns an
        empty list for HDF5/MAT files (use an HDF5 browser instead).
        """
        if self.fmt not in self._VTK_EXTENSIONS:
            return []
        try:
            import pyvista as pv
        except ImportError:
            return []
        mesh = pv.read(self.path)
        return list(mesh.point_data.keys()) + list(mesh.cell_data.keys())

    def __repr__(self) -> str:
        return (
            f"DVC(n_steps={self.n_steps}, step={self._step}, "
            f"shape={self.shape}, spacing={self.spacing})"
        )

    # ------------------------------------------------------------------
    # Step loader dispatcher
    # ------------------------------------------------------------------

    def _load_step(self, step: int) -> None:
        if self._mode == "vtk_single":
            self._load_vtk(self.path)

        elif self._mode == "vtk_multi":
            self._load_vtk(self._paths[step])

        elif self._mode == "h5_single":
            self._load_h5(self.path, self._displacement_key)

        elif self._mode == "h5_5d":
            self._load_h5_5d(step)

        elif self._mode == "h5_group":
            key = self._step_keys[step]
            full_key = f"{key}/{self._displacement_key}"
            self._load_h5(self.path, full_key)

        self.shape: tuple[int, ...] = self.displacement.shape[:3]

    # ------------------------------------------------------------------
    # VTK loaders
    # ------------------------------------------------------------------

    def _read_vtk_geometry(self, path: str) -> None:
        """Read origin and spacing from a VTK file without loading fields."""
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "pyvista is required to load VTK files: pip install pyvista"
            ) from e
        mesh = pv.read(path)
        self._set_geometry_from_vtk(mesh)

    def _load_vtk(self, path: str) -> None:
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "pyvista is required to load VTK files: pip install pyvista"
            ) from e

        mesh = pv.read(path)
        self._set_geometry_from_vtk(mesh)
        dims = _vtk_dims(mesh)

        self.displacement = _extract_field(
            mesh, self._displacement_key, dims, components=3)

        self.strain = {}
        if self._strain_keys:
            for key in self._strain_keys:
                try:
                    n_comp = _field_n_components(mesh, key)
                    self.strain[key] = _extract_field(mesh, key, dims,
                                                      components=n_comp)
                except KeyError:
                    warnings.warn(f"Strain field '{key}' not found in {path}.")

    def _set_geometry_from_vtk(self, mesh) -> None:
        if hasattr(mesh, "origin") and hasattr(mesh, "spacing"):
            self.origin  = tuple(float(v) for v in mesh.origin)
            self.spacing = tuple(float(v) for v in mesh.spacing)
        elif hasattr(mesh, "bounds"):
            b = mesh.bounds
            self.origin  = (float(b[0]), float(b[2]), float(b[4]))
            self.spacing = (1.0, 1.0, 1.0)

    # ------------------------------------------------------------------
    # HDF5 loaders
    # ------------------------------------------------------------------

    def _probe_h5(self, path: str, key: str) -> tuple[str, int]:
        """Determine HDF5 mode and n_steps without loading the full array."""
        import h5py

        with h5py.File(path, "r") as f:
            shape = f[key].shape

        if len(shape) == 5:                   # (n_steps, nx, ny, nz, 3)
            import h5py
            with h5py.File(path, "r") as f:
                self._h5_all_steps = f[key][:]   # load once; fits in RAM
            self._read_h5_geometry(path)
            return "h5_5d", shape[0]

        self._read_h5_geometry(path)
        return "h5_single", 1

    def _read_h5_geometry(self, path: str) -> None:
        import h5py

        with h5py.File(path, "r") as f:
            if self._origin_override is not None:
                self.origin = tuple(float(v) for v in self._origin_override)
            elif self._origin_key and self._origin_key in f:
                self.origin = tuple(float(v) for v in f[self._origin_key][:])

            if self._spacing_override is not None:
                self.spacing = tuple(float(v) for v in self._spacing_override)
            elif self._spacing_key and self._spacing_key in f:
                self.spacing = tuple(float(v) for v in f[self._spacing_key][:])

    def _load_h5(self, path: str, displacement_key: str) -> None:
        import h5py

        with h5py.File(path, "r") as f:
            self.displacement = f[displacement_key][:]
            self._read_h5_geometry(path)

    def _load_h5_5d(self, step: int) -> None:
        self.displacement = self._h5_all_steps[step]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _vtk_dims(mesh) -> tuple[int, ...]:
    if hasattr(mesh, "dimensions"):
        return tuple(int(d) for d in mesh.dimensions)
    return (mesh.n_points,)


def _field_n_components(mesh, key: str) -> int:
    if key in mesh.point_data:
        arr = mesh.point_data[key]
    elif key in mesh.cell_data:
        arr = mesh.cell_data[key]
    else:
        raise KeyError(key)
    return 1 if arr.ndim == 1 else arr.shape[1]


def _extract_field(
    mesh, key: str, dims: tuple[int, ...], components: int
) -> np.ndarray:
    if key in mesh.point_data:
        raw = np.array(mesh.point_data[key])
        grid_dims = dims
    elif key in mesh.cell_data:
        raw = np.array(mesh.cell_data[key])
        grid_dims = tuple(max(d - 1, 1) for d in dims)
    else:
        raise KeyError(
            f"Field '{key}' not found. "
            f"Available: {list(mesh.point_data.keys()) + list(mesh.cell_data.keys())}"
        )
    if components == 1:
        return raw.reshape(grid_dims)
    return raw.reshape(grid_dims + (components,))

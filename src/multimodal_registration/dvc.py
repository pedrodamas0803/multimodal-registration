"""
DVC (Digital Volume Correlation) results loader.

Two classes are provided:

**DVC** — regular-grid displacement field
    Supports VTK (``.vtk``, ``.vts``, ``.vtu``) and HDF5/MATLAB
    (``.mat``, ``.h5``, ``.hdf5``) files.  Three HDF5 layouts are
    recognised automatically (single-step, multi-group, 5-D array).

**DVCMesh** — unstructured finite-element mesh (MATLAB v7.3 / HDF5)
    Reads MATLAB DVC result files that store nodal positions in ``xo``,
    ``yo``, ``zo`` and displacements in ``U``, with axis 0 being the
    loading step (as written by MATLAB).  Also reads element connectivity,
    mesh metadata, and the ``param`` parameter group.

Use :meth:`DVC.select_step` / :meth:`DVCMesh.select_step` to switch
between loading steps.

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
    _H5_EXTENSIONS  = {"mat", "h5", "hdf5", "res"}

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


# ---------------------------------------------------------------------------
# DVCMesh — unstructured mesh reader (MATLAB v7.3 / HDF5)
# ---------------------------------------------------------------------------

class DVCMesh:
    """DVC results on an unstructured finite-element mesh (MATLAB v7.3 / HDF5).

    Reads MATLAB DVC result files with the following top-level layout:

    =========  ================================================  ======================================
    Dataset    h5py shape                                        Description
    =========  ================================================  ======================================
    xo         ``(n_steps, n_nodes)``                            Nodal x-coordinates per step
    yo         ``(n_steps, n_nodes)``                            Nodal y-coordinates per step
    zo         ``(n_steps, n_nodes)``                            Nodal z-coordinates per step
    U          ``(n_steps, …)``                                  Nodal displacements per step
    conn       MATLAB ``[n_elems, 8]`` → h5py ``(8, n_elems)``  Element connectivity (hex8)
    Nelems     scalar                                            Number of elements
    Nnodes     scalar                                            Number of nodes
    ns         scalar                                            Number of steps
    ng         scalar                                            Gauss points per element (8 for hex8)
    nmod       scalar                                            Number of modes
    rint       ``(n_steps, n_elems, n_gauss)``                   Integration-point results
    param      group                                             Analysis parameters
    =========  ================================================  ======================================

    Elements are 8-node trilinear hexahedra (hex8).  ``conn`` is stored by
    MATLAB as ``[n_elems × 8]`` (row = element, column = local node index),
    which h5py reads transposed as ``(8, n_elems)``; the reader corrects this
    so ``connectivity`` is always ``(n_elems, 8)``.

    After construction ``nodes`` and ``displacement`` reflect the active step
    in **pixel** coordinates.  Use ``nodes_phys`` for physical coordinates.

    Parameters
    ----------
    path:
        Path to the MATLAB v7.3 / HDF5 result file (``.mat``, ``.h5``).
    step:
        Initial active step index (0-based).  Negative values count from
        the end (default ``-1`` → last step).
    pixel_size:
        Physical size of one voxel.  When *None*, the value is read from
        ``param/pixel_size`` in the file; falls back to ``1.0`` if absent.
        All visualisation methods and coordinate queries use physical units
        (pixels × pixel_size).
    unit:
        Label appended to axis annotations in plots, e.g. ``'μm'``,
        ``'mm'``.  Default ``'px'``.
    """

    def __init__(
        self,
        path: str,
        step: int = -1,
        pixel_size: float | None = None,
        unit: str = "px",
    ):
        import h5py

        self.path = path

        with h5py.File(path, "r") as f:
            # Per-axis counts — Nnodes/Nelems shape (3, 1), e.g. [56, 24, 76]
            _nnodes = np.array(f["Nnodes"][()], dtype=int).ravel()
            _nelems = np.array(f["Nelems"][()], dtype=int).ravel()
            self.n_nodes_per_axis = tuple(_nnodes.tolist())
            self.n_elems_per_axis = tuple(_nelems.tolist())
            self.n_nodes = int(_nnodes.prod())
            self.n_elems = int(_nelems.prod())
            self.nodes_per_elem = 8   # hex8 — trilinear hexahedra
            # n_gauss fixed at 8 for hex8 regardless of what ng stores in the file
            self.n_gauss = 8

            self.params = _read_mat_params(f["param"]) if "param" in f else {}

            # Resolve pixel size: explicit arg > param group > default 1
            if pixel_size is not None:
                self.pixel_size = float(pixel_size)
            elif "pixel_size" in self.params:
                self.pixel_size = float(self.params["pixel_size"])
            else:
                self.pixel_size = 1.0
            self.unit = unit

            # Expose every param field as a direct attribute so callers can
            # write dvc.pixel_size, dvc.roi, dvc.analysis, etc.
            # Existing attributes (n_nodes, pixel_size, …) are never overwritten.
            for _k, _v in self.params.items():
                if _k.isidentifier() and not hasattr(self, _k):
                    setattr(self, _k, _v)

            # xo/yo/zo hold the reference (undeformed) nodal coordinates.
            # Shape in file: (1, n_nodes) — squeeze to (n_nodes,).
            xo_ref = np.array(f["xo"]).ravel()
            yo_ref = np.array(f["yo"]).ravel()
            zo_ref = np.array(f["zo"]).ravel()

            # U: (n_steps, 3·n_nodes).  DOFs grouped by component:
            #   [ux_0…ux_{N-1}, uy_0…uy_{N-1}, uz_0…uz_{N-1}]
            # n_steps is read from U directly — ns in the file is per-axis metadata.
            U_raw = np.array(f["U"])   # (n_steps, 3·n_nodes)

            # Connectivity: MATLAB [n_elems, 8] → h5py (8, n_elems); 0-based.
            if "conn" in f:
                conn = np.array(f["conn"], dtype=int)
                conn = conn.T if conn.shape == (8, self.n_elems) else conn
                self.connectivity = conn - 1
            else:
                self.connectivity = None

            # Smesh and rint are stored as-is (metadata / scalar flags).
            self.Smesh = np.array(f["Smesh"]) if "Smesh" in f else None
            self.rint  = np.array(f["rint"])  if "rint"  in f else None

        # Reference positions: (n_nodes, 3)
        self._ref_nodes = np.stack([xo_ref, yo_ref, zo_ref], axis=-1)

        # n_steps comes from U (ns in the file is per-axis mesh metadata)
        self.n_steps = U_raw.shape[0]

        # Reshape U: grouped DOFs → (n_steps, n_nodes, 3)
        self._U_all = U_raw.reshape(self.n_steps, 3, self.n_nodes).transpose(0, 2, 1)

        # Activate requested step
        idx = step if step >= 0 else self.n_steps + step
        if not 0 <= idx < self.n_steps:
            raise IndexError(
                f"step={step} out of range for n_steps={self.n_steps}."
            )
        self._step = idx
        self._activate(idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Currently active step index (0-based)."""
        return self._step

    @property
    def nodes_phys(self) -> np.ndarray:
        """Current-step node coordinates in physical units ``(n_nodes, 3)``.

        Equivalent to ``nodes * pixel_size``.
        """
        return self.nodes * self.pixel_size

    def select_step(self, step: int) -> None:
        """Switch to a different loading step.

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
        self._activate(step)

    def __repr__(self) -> str:
        return (
            f"DVCMesh(path='{self.path}', n_steps={self.n_steps}, "
            f"step={self._step}, n_nodes={self.n_nodes}, "
            f"n_elems={self.n_elems}, pixel_size={self.pixel_size} {self.unit})"
        )

    # ------------------------------------------------------------------
    # Strain and invariants
    # ------------------------------------------------------------------

    def compute_strain(self) -> np.ndarray:
        """Compute the small-strain tensor at all Gauss points for the current step.

        Uses trilinear hex8 shape functions and a 2×2×2 Gauss rule.

        Returns
        -------
        strain : ndarray, shape ``(n_elems, n_gauss, 6)``
            Voigt components ``[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]``.
            Also stored as ``self.strain``.
        """
        elem_nodes = self.nodes[self.connectivity]        # (n_elems, 8, 3)
        elem_disp  = self.displacement[self.connectivity] # (n_elems, 8, 3)

        # Jacobian at every (element, Gauss point): J[e,g,i,j] = ∂x_j/∂ξ_i
        J = np.einsum('gki,ekj->egij', _HEX8_dN_NAT, elem_nodes)
        J_inv = np.linalg.inv(J)                          # (n_elems, n_gauss, 3, 3)

        # Physical shape-function derivatives: dN[e,g,j,k] = ∂N_k/∂x_j
        dN = np.einsum('egji,gki->egjk', J_inv, _HEX8_dN_NAT)

        # Displacement gradient: H[e,g,i,j] = ∂u_i/∂x_j
        H = np.einsum('egjk,eki->egij', dN, elem_disp)

        # Symmetric small-strain tensor
        eps = 0.5 * (H + H.transpose(0, 1, 3, 2))        # (n_elems, n_gauss, 3, 3)

        self.strain = np.stack([
            eps[..., 0, 0], eps[..., 1, 1], eps[..., 2, 2],
            eps[..., 0, 1], eps[..., 0, 2], eps[..., 1, 2],
        ], axis=-1)                                        # (n_elems, n_gauss, 6)
        return self.strain

    def compute_invariants(self, strain: np.ndarray | None = None) -> dict:
        """Compute scalar strain invariants from a Voigt strain array.

        Parameters
        ----------
        strain : ndarray ``(…, 6)``, optional
            Voigt components ``[ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz]``.
            Uses ``self.strain`` (calling :meth:`compute_strain` first if
            needed) when omitted.

        Returns
        -------
        dict with keys:

        ``volumetric``
            I₁ = tr(ε) = ε_xx + ε_yy + ε_zz, shape ``(…)``.
        ``von_mises``
            Equivalent strain √(⅔ ε_dev:ε_dev), shape ``(…)``.

        Also stored as ``self.invariants``.
        """
        if strain is None:
            if not hasattr(self, "strain"):
                self.compute_strain()
            strain = self.strain

        self.invariants = _invariants_from_voigt(strain)
        return self.invariants

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_pyvista(self, component: str | None = None):
        """Build a PyVista ``UnstructuredGrid`` for the current step.

        The mesh is built from ``self.nodes`` and ``self.connectivity``
        (hex8 elements, VTK cell type 12).  If *component* is given,
        the Gauss-point-averaged strain scalar is attached as cell data.

        Parameters
        ----------
        component : str, optional
            Strain component to attach (same choices as
            :meth:`plot_strain_field`).

        Returns
        -------
        pyvista.UnstructuredGrid
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "pyvista is required for mesh visualisation: pip install pyvista"
            ) from exc

        prefix = np.full((self.n_elems, 1), 8, dtype=np.int64)
        cells = np.hstack([prefix, self.connectivity]).ravel()
        celltypes = np.full(self.n_elems, 12, dtype=np.uint8)   # VTK_HEXAHEDRON

        grid = pv.UnstructuredGrid(cells, celltypes, self.nodes_phys)
        if component is not None:
            grid.cell_data[component] = self._strain_cell_data(component)
        return grid

    def plot_strain_field(
        self,
        component: str = "von_mises",
        step: int | None = None,
        show_edges: bool = False,
        clim: tuple | None = None,
        plotter=None,
        **kwargs,
    ):
        """3-D interactive visualisation of a strain field using PyVista.

        Parameters
        ----------
        component : str
            Strain component to display (same choices as
            :meth:`plot_strain_history`).
        step : int, optional
            Step to display; switches the active step when given.
        show_edges : bool
            Draw element edges (default *False*).
        clim : tuple ``(vmin, vmax)``, optional
            Colour-map limits.
        plotter : pyvista.Plotter, optional
            Existing plotter to add the mesh to; a new one is created
            when *None*.
        **kwargs
            Forwarded to ``pv.Plotter.add_mesh()``.

        Returns
        -------
        pyvista.Plotter
            Call ``.show()`` to open the interactive window.
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "pyvista is required: pip install pyvista"
            ) from exc

        if step is not None:
            self.select_step(step)
        self.compute_strain()
        grid = self.to_pyvista(component=component)

        pl = plotter if plotter is not None else pv.Plotter()
        pl.add_mesh(
            grid,
            scalars=component,
            show_edges=show_edges,
            clim=clim,
            scalar_bar_args={"title": component},
            **kwargs,
        )
        return pl

    def plot_strain_slice(
        self,
        component: str = "von_mises",
        normal: str | int = "z",
        origin: float | None = None,
        thickness: float | None = None,
        step: int | None = None,
        cmap: str = "jet",
        ax=None,
        pct=None,
        pct_alpha: float = 0.5,
        **kwargs,
    ):
        """2-D cross-section of the strain field using matplotlib.

        Elements whose centroid falls within *thickness*/2 of the slice
        plane are projected onto the plane and displayed as a filled
        triangulation.

        Parameters
        ----------
        component : str
            Strain component to display (same choices as
            :meth:`plot_strain_history`).
        normal : ``'x'``, ``'y'``, ``'z'`` or int index
            Normal direction of the slice plane.
        origin : float, optional
            Position of the plane along *normal* in physical units; defaults
            to the mesh mid-point.
        thickness : float, optional
            Slab half-width used to select elements.  Defaults to ~1.5×
            the estimated element size along *normal*.
        step : int, optional
            Step to display; switches the active step when given.
        cmap : str
            Matplotlib colour map (default ``'jet'``).
        ax : matplotlib Axes, optional
            Existing axes; a new figure is created when *None*.
        pct : ReferencePCT, optional
            Reference PCT volume.  When supplied, a grayscale slice is drawn
            as a fully opaque background and the strain overlay is rendered
            with transparency *pct_alpha*.

            Axis convention assumed:

            * ``pct.vol`` shape ``(Nz, Ny, Nx)`` — PCT axis 0 = Z, 1 = Y, 2 = X.
            * DVC node coordinates (xo, yo, zo) are voxel indices into that
              same volume (DVC x → PCT axis 2, DVC y → 1, DVC z → 0).
        pct_alpha : float
            Opacity of the strain overlay when *pct* is given (0 = fully
            transparent, 1 = fully opaque).  Default ``0.5``.
        **kwargs
            Forwarded to ``ax.tripcolor()``.

        Returns
        -------
        fig, ax, mappable
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        _AXES = {"x": 0, "y": 1, "z": 2}
        ni = _AXES[normal.lower()] if isinstance(normal, str) else int(normal)
        h0, h1 = (ni + 1) % 3, (ni + 2) % 3
        axis_names = ["x", "y", "z"]

        if step is not None:
            self.select_step(step)
        vals = self._strain_cell_data(component)           # (n_elems,)

        centroids = self.nodes_phys[self.connectivity].mean(axis=1)  # (n_elems, 3)
        cn = centroids[:, ni]

        if origin is None:
            origin = float(cn.mean())
        if thickness is None:
            extent = float(cn.ptp())
            n_layers = max(round(self.n_elems ** (1 / 3)), 1)
            thickness = 1.5 * extent / n_layers

        mask = np.abs(cn - origin) <= thickness / 2
        if mask.sum() == 0:
            raise ValueError(
                f"No elements within slice at {axis_names[ni]}={origin:.3g} {self.unit}"
                f" ± {thickness/2:.3g}.  Increase thickness or adjust origin."
            )

        x = centroids[mask, h0]
        y = centroids[mask, h1]
        z = vals[mask]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # --- PCT grayscale background ----------------------------------------
        if pct is not None:
            px = self.pixel_size
            # DVC axis i → PCT axis (2 - i):  DVC x(0)→PCT 2, y(1)→1, z(2)→0
            pct_ni = 2 - ni
            slice_idx = int(np.clip(
                round(origin / px), 0, pct.vol.shape[pct_ni] - 1
            ))
            pct_slice = np.take(pct.vol, slice_idx, axis=pct_ni)  # 2D

            # np.take leaves remaining PCT axes in sorted order.
            # Map each remaining PCT axis back to its DVC axis (2 - pct_ax).
            # If the first remaining dimension corresponds to h0, the array is
            # (n_h0, n_h1) and must be transposed to (n_h1, n_h0) for imshow.
            remaining_pct = sorted(a for a in range(3) if a != pct_ni)
            if (2 - remaining_pct[0]) == h0:
                pct_slice = pct_slice.T

            # Physical extent along h0 (horizontal) and h1 (vertical)
            n_h0 = pct.vol.shape[2 - h0]
            n_h1 = pct.vol.shape[2 - h1]
            extent_pct = [0, n_h0 * px, 0, n_h1 * px]
            ax.imshow(
                pct_slice, cmap="gray", extent=extent_pct,
                origin="lower", aspect="auto",
            )
            # Strain drawn on top with reduced opacity
            kwargs.setdefault("alpha", pct_alpha)

        # --- Strain overlay --------------------------------------------------
        try:
            tri = mtri.Triangulation(x, y)
            mappable = ax.tripcolor(tri, z, cmap=cmap, **kwargs)
        except Exception:
            mappable = ax.scatter(x, y, c=z, cmap=cmap, **kwargs)

        unit_label = f" [{self.unit}]" if self.unit else ""
        fig.colorbar(mappable, ax=ax, label=component)
        ax.set_xlabel(f"{axis_names[h0]}{unit_label}")
        ax.set_ylabel(f"{axis_names[h1]}{unit_label}")
        ax.set_title(
            f"Strain ({component})  |  "
            f"{axis_names[ni]} = {origin:.3g}{unit_label}  |  step {self._step}"
        )
        ax.set_aspect("equal")

        return fig, ax, mappable

    def plot_strain_history(
        self,
        coords,
        component: str = "von_mises",
        ref_step: int = 0,
        ax=None,
    ):
        """Plot strain history at the element(s) nearest to given coordinate(s).

        Strain is computed in a single vectorised pass over all steps — the
        Jacobian is built once from the reference configuration for the target
        elements only, then the displacement gradient is evaluated for every
        step simultaneously with one einsum.

        Parameters
        ----------
        coords : array-like ``(3,)`` or list of ``(3,)``
            Physical coordinate(s) ``(x, y, z)`` to probe.  Each point is
            matched to the nearest element centroid at *ref_step*.
        component : str
            ``'xx'``, ``'yy'``, ``'zz'``, ``'xy'``, ``'xz'``, ``'yz'``
            for Voigt strain components; ``'von_mises'`` for equivalent
            strain; ``'volumetric'`` for the trace.
        ref_step : int
            Step used to locate element centroids (default 0).
        ax : matplotlib Axes, optional
            Existing axes to draw on; a new figure is created if *None*.

        Returns
        -------
        fig, ax, values
            ``values`` has shape ``(n_coords, n_steps)``.
        """
        import matplotlib.pyplot as plt

        coords = np.atleast_2d(coords)          # (n_coords, 3)
        n_coords = len(coords)

        ref_nodes_phys = (self._ref_nodes + self._U_all[ref_step]) * self.pixel_size
        centroids = ref_nodes_phys[self.connectivity].mean(axis=1)
        elem_indices = np.array([
            int(np.argmin(np.linalg.norm(centroids - c, axis=1)))
            for c in coords
        ])

        # Compute strain for target elements across ALL steps in one pass:
        # → (n_steps, n_coords, n_gauss, 6)
        strain_hist = self._strain_history_at_elems(elem_indices)

        _VOIGT = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "xz": 4, "yz": 5}
        if component in _VOIGT:
            # (n_steps, n_coords, n_gauss) → mean over gauss → (n_coords, n_steps)
            values = strain_hist[..., _VOIGT[component]].mean(axis=2).T
        else:
            inv = _invariants_from_voigt(strain_hist)
            if component not in inv:
                raise ValueError(
                    f"Unknown component '{component}'. "
                    f"Choose from: {sorted(_VOIGT) + ['von_mises', 'volumetric']}"
                )
            values = inv[component].mean(axis=2).T  # (n_coords, n_steps)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        steps = np.arange(self.n_steps)
        for ci, (c, ei) in enumerate(zip(coords, elem_indices)):
            label = f"({c[0]:.2g}, {c[1]:.2g}, {c[2]:.2g})  [elem {ei}]"
            ax.plot(steps, values[ci], "o-", label=label)

        ax.set_xlabel("Step")
        ax.set_ylabel(f"Strain — {component}")
        if n_coords > 1:
            ax.legend()

        return fig, ax, values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _activate(self, step: int) -> None:
        """Set ``displacement`` and deformed ``nodes`` for the given step."""
        self.displacement: np.ndarray = self._U_all[step]                      # (n_nodes, 3)
        self.nodes: np.ndarray = self._ref_nodes + self.displacement            # (n_nodes, 3)

    def _strain_history_at_elems(self, elem_indices: np.ndarray) -> np.ndarray:
        """Voigt strain for specific elements across all steps in one pass.

        The Jacobian is built once from the reference configuration — valid for
        the small-strain regime typical of DVC.  Displacements for all steps
        are then contracted in a single einsum so there is no Python loop over
        steps.

        Parameters
        ----------
        elem_indices : ndarray ``(n_target,)``

        Returns
        -------
        strain : ndarray ``(n_steps, n_target, n_gauss, 6)``
            Voigt components ``[εxx, εyy, εzz, εxy, εxz, εyz]``.
        """
        conn          = self.connectivity[elem_indices]      # (n_target, 8)
        ref_elem_nodes = self._ref_nodes[conn]               # (n_target, 8, 3)

        # Jacobian and physical shape-function derivatives — computed once
        J     = np.einsum('gki,ekj->egij', _HEX8_dN_NAT, ref_elem_nodes)
        J_inv = np.linalg.inv(J)                             # (n_target, n_gauss, 3, 3)
        dN    = np.einsum('egji,gki->egjk', J_inv, _HEX8_dN_NAT)  # (n_target, n_gauss, 3, 8)

        # Displacements for every step at target-element nodes: (n_steps, n_target, 8, 3)
        disp_all = self._U_all[:, conn, :]

        # Displacement gradient for all steps at once: (n_steps, n_target, n_gauss, 3, 3)
        # H[s,e,g,i,j] = ∂u_i/∂x_j = Σ_k dN[e,g,j,k] · disp[s,e,k,i]
        H   = np.einsum('egjk,seki->segij', dN, disp_all)
        eps = 0.5 * (H + H.transpose(0, 1, 2, 4, 3))

        return np.stack([
            eps[..., 0, 0], eps[..., 1, 1], eps[..., 2, 2],
            eps[..., 0, 1], eps[..., 0, 2], eps[..., 1, 2],
        ], axis=-1)                                          # (n_steps, n_target, n_gauss, 6)

    def _strain_cell_data(self, component: str) -> np.ndarray:
        """Return Gauss-point-averaged strain scalar per element ``(n_elems,)``."""
        if not hasattr(self, "strain"):
            self.compute_strain()
        _VOIGT = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "xz": 4, "yz": 5}
        if component in _VOIGT:
            return self.strain[:, :, _VOIGT[component]].mean(axis=1)
        inv = _invariants_from_voigt(self.strain)
        if component not in inv:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Choose from: {sorted(_VOIGT) + sorted(inv)}"
            )
        return inv[component].mean(axis=1)


# ---------------------------------------------------------------------------
# DVCMesh module-level helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Hex8 element constants (trilinear hexahedra, 2×2×2 Gauss rule)
# ---------------------------------------------------------------------------

# Natural coordinates of the 8 nodes: (ξ, η, ζ) ∈ {-1, +1}³
_HEX8_NODES = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=float)

# 2×2×2 Gauss points and weights (weight = 1 for all, total = 8)
_g = 1.0 / np.sqrt(3)
_HEX8_GAUSS_PTS = np.array([
    [-_g, -_g, -_g], [ _g, -_g, -_g], [ _g,  _g, -_g], [-_g,  _g, -_g],
    [-_g, -_g,  _g], [ _g, -_g,  _g], [ _g,  _g,  _g], [-_g,  _g,  _g],
])
def _hex8_shape_deriv(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Shape-function derivatives in natural coords at one point.

    Returns array of shape ``(8, 3)``: ``dN[i, j] = ∂N_i/∂ξ_j``.
    """
    xn, en, zn = _HEX8_NODES[:, 0], _HEX8_NODES[:, 1], _HEX8_NODES[:, 2]
    dN_dxi   = (1/8) * xn * (1 + en * eta)  * (1 + zn * zeta)
    dN_deta  = (1/8) * en * (1 + xn * xi)   * (1 + zn * zeta)
    dN_dzeta = (1/8) * zn * (1 + xn * xi)   * (1 + en * eta)
    return np.stack([dN_dxi, dN_deta, dN_dzeta], axis=1)  # (8, 3)


# Precompute for all 8 Gauss points: shape (n_gauss=8, n_nodes=8, 3)
_HEX8_dN_NAT = np.stack([_hex8_shape_deriv(*gp) for gp in _HEX8_GAUSS_PTS])


def _invariants_from_voigt(s: np.ndarray) -> dict:
    """Compute strain invariants from a Voigt array ``(…, 6)``.

    Returns a dict with ``'volumetric'`` (I₁ = tr ε) and ``'von_mises'``
    (equivalent strain √(⅔ ε_dev:ε_dev)), both of shape ``(…)``.
    """
    exx, eyy, ezz = s[..., 0], s[..., 1], s[..., 2]
    exy, exz, eyz = s[..., 3], s[..., 4], s[..., 5]
    I1 = exx + eyy + ezz
    h  = I1 / 3
    vm = np.sqrt(np.maximum(0.0,
        2/3 * ((exx - h)**2 + (eyy - h)**2 + (ezz - h)**2
               + 2 * (exy**2 + exz**2 + eyz**2))
    ))
    return {"volumetric": I1, "von_mises": vm}


def _read_mat_params(group) -> dict:
    """Read a flat HDF5 group (MATLAB ``param`` struct) into a Python dict.

    Handles:
    * Numeric scalars and arrays
    * Byte / variable-length / fixed-length HDF5 strings
    * MATLAB char arrays stored as ``uint16`` Unicode code points
    * HDF5 object references — dereferenced via the root file (MATLAB stores
      long strings and cell arrays in the ``#refs#`` group)
    """
    import h5py

    params = {}
    for key in group.keys():
        try:
            val = group[key][()]
        except Exception:
            continue

        # --- resolve HDF5 object references --------------------------------
        if isinstance(val, h5py.Reference):
            val = group.file[val][()]

        elif isinstance(val, np.ndarray) and val.dtype == object:
            refs = val.ravel()
            if len(refs) and all(isinstance(r, h5py.Reference) for r in refs):
                # Array of references → decode each chunk, join as string
                chunks = []
                for r in refs:
                    chunk = group.file[r][()]
                    if isinstance(chunk, np.ndarray) and chunk.dtype == np.uint16:
                        chunks.append("".join(chr(c) for c in chunk.ravel() if c != 0))
                    elif isinstance(chunk, bytes):
                        chunks.append(chunk.decode("utf-8", errors="replace"))
                    else:
                        chunks.append(str(np.squeeze(chunk)))
                val = "".join(chunks)

        # --- decode string encodings ----------------------------------------
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        elif isinstance(val, np.ndarray):
            if val.dtype.kind in ("S", "U"):
                flat = val.flat[0]
                val = flat.decode("utf-8", errors="replace") if isinstance(flat, bytes) else flat
            elif val.dtype == np.uint16:
                # MATLAB char array (UTF-16 code units)
                val = "".join(chr(c) for c in val.ravel() if c != 0)
            elif val.size == 1:
                val = val.flat[0]

        params[key] = val
    return params

"""
Export DCT, PCT and DVC data to HDF5, VTK and DREAM.3D files.

Functions
---------
write_h5        — Combined HDF5 with /DCT, /PCT, /DVC groups.
write_vtk       — One ImageData VTK file per dataset (DCT and/or DVC).
write_dream3d   — DREAM.3D 6.x compatible HDF5 for DREAM.3D / ParaView.
write           — Dispatcher: infers format from the file extension.

DREAM.3D notes
--------------
The DREAM.3D writer produces a file compatible with DREAM.3D 6.x / SIMPL.
It requires ``dct.EulerAngle`` (per-grain, Bunge ZXZ) and ``dct.GIDvol``
to build the per-voxel orientation and phase arrays expected by DREAM.3D.
Euler angles are written in **radians** regardless of how they are stored
in the DCT object (use ``euler_degrees=True``, the default, if they are in
degrees).

VTK notes
---------
Requires **pyvista**: ``pip install "multimodal-registration[vtk]"``.
Arrays are written as point data on an ImageData (uniform rectilinear)
grid.  Spatial axes follow the VTK/Fortran convention (x changes fastest).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dct import DCT
from .dvc import DVC
from .pct import ReferencePCT

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vtk_flat_scalar(arr: np.ndarray) -> np.ndarray:
    """Flatten a (nx, ny, nz) array to VTK Fortran order."""
    return arr.ravel(order="F")


def _vtk_flat_vector(arr: np.ndarray) -> np.ndarray:
    """Flatten a (nx, ny, nz, nc) array to VTK Fortran order, shape (N, nc)."""
    nx, ny, nz, nc = arr.shape
    # Transpose spatial dims so C-order ravel gives x-fastest (VTK convention)
    return arr.transpose(2, 1, 0, 3).reshape(nx * ny * nz, nc)


def _dream3d_flat_scalar(arr: np.ndarray) -> np.ndarray:
    """Flatten (nx, ny, nz) → (N, 1) in DREAM.3D tuple order (x-fastest)."""
    return arr.ravel(order="F").reshape(-1, 1)


def _dream3d_flat_vector(arr: np.ndarray) -> np.ndarray:
    """Flatten (nx, ny, nz, nc) → (N, nc) in DREAM.3D tuple order."""
    return _vtk_flat_vector(arr)          # same convention


def _set_dream3d_attrs(ds, component_dims: list[int], object_type: str,
                       tuple_dims: list[int]) -> None:
    """Write the three mandatory DREAM.3D attributes on a dataset."""
    ds.attrs["ComponentDimensions"] = np.array(component_dims, dtype=np.uint64)
    ds.attrs["ObjectType"]          = np.bytes_(object_type)
    ds.attrs["TupleDimensions"]     = np.array(tuple_dims,     dtype=np.uint64)


_DTYPE_TO_DREAM3D: dict[str, str] = {
    "int8":    "DataArray<int8_t>",
    "int16":   "DataArray<int16_t>",
    "int32":   "DataArray<int32_t>",
    "int64":   "DataArray<int64_t>",
    "uint8":   "DataArray<uint8_t>",
    "uint16":  "DataArray<uint16_t>",
    "uint32":  "DataArray<uint32_t>",
    "uint64":  "DataArray<uint64_t>",
    "float32": "DataArray<float>",
    "float64": "DataArray<double>",
    "bool":    "DataArray<bool>",
}


def _dream3d_type(arr: np.ndarray) -> str:
    return _DTYPE_TO_DREAM3D.get(arr.dtype.name, "DataArray<float>")


# ---------------------------------------------------------------------------
# HDF5 export
# ---------------------------------------------------------------------------

def write_h5(
    path: str,
    dct: DCT | None = None,
    pct: ReferencePCT | None = None,
    dvc: DVC | None = None,
) -> None:
    """Write DCT, PCT and/or DVC data to a single HDF5 file.

    The file mirrors the original input layout so that it can be passed
    back to the loaders:

    .. code-block:: text

        /DCT/DS/<key>          — all arrays loaded from the DCT HDF5
        /PCT/vol               — PCT grey-level volume
        /PCT/mask              — Otsu binary mask
        /DVC/displacement      — displacement field (nx, ny, nz, 3)
        /DVC/strain/<key>      — strain fields (if present)
        /DVC/origin            — ROI origin in PCT voxel space
        /DVC/spacing           — DVC grid spacing

    Parameters
    ----------
    path:
        Output file path (``*.h5`` or ``*.hdf5``).
    dct:
        DCT object to export.
    pct:
        PCT reference object to export.
    dvc:
        DVC object to export.
    """
    import h5py

    with h5py.File(path, "w") as hout:
        if dct is not None:
            grp = hout.require_group("DCT/DS")
            grp.attrs["voxel_size"] = dct.voxel_size
            for key in vars(dct):
                if key.startswith("_"):
                    continue
                val = getattr(dct, key)
                if isinstance(val, np.ndarray):
                    grp.create_dataset(key, data=val, compression="gzip",
                                       compression_opts=4)

        if pct is not None:
            grp = hout.require_group("PCT")
            grp.create_dataset("vol",  data=pct.vol,  compression="gzip",
                               compression_opts=4)
            grp.create_dataset("mask", data=pct.mask, compression="gzip",
                               compression_opts=4)

        if dvc is not None:
            grp = hout.require_group("DVC")
            grp.create_dataset("displacement", data=dvc.displacement,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("origin",  data=np.array(dvc.origin))
            grp.create_dataset("spacing", data=np.array(dvc.spacing))
            if dvc.strain:
                sgrp = grp.require_group("strain")
                for key, arr in dvc.strain.items():
                    sgrp.create_dataset(key, data=arr, compression="gzip",
                                        compression_opts=4)


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def write_vtk(
    path: str,
    dct: DCT | None = None,
    dvc: DVC | None = None,
) -> None:
    """Write DCT and/or DVC volumes as VTK ImageData files.

    When both *dct* and *dvc* are supplied, two files are written:
    ``<stem>_dct.vts`` and ``<stem>_dvc.vts``.  When only one is given,
    a single file is written at *path*.

    Requires **pyvista** (``pip install "multimodal-registration[vtk]"``).

    Parameters
    ----------
    path:
        Output file path.  The suffix determines the VTK format
        (``.vts`` structured grid recommended).
    dct:
        DCT object to export.
    dvc:
        DVC object to export.
    """
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "pyvista is required for VTK export: "
            "pip install \"multimodal-registration[vtk]\""
        ) from e

    p = Path(path)
    write_both = dct is not None and dvc is not None

    if dct is not None:
        out_path = str(p.with_stem(p.stem + "_dct")) if write_both else path
        _write_dct_vtk(dct, out_path, pv)

    if dvc is not None:
        out_path = str(p.with_stem(p.stem + "_dvc")) if write_both else path
        _write_dvc_vtk(dvc, out_path, pv)


def _write_dct_vtk(dct: DCT, path: str, pv) -> None:
    nx, ny, nz = dct.shape
    n = nx * ny * nz

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing    = (dct.voxel_size,) * 3
    grid.origin     = (0.0, 0.0, 0.0)

    for key in dct._vol_keys:
        arr = getattr(dct, key)
        if arr.ndim == 3:
            grid.point_data[key] = _vtk_flat_scalar(arr)
        elif arr.ndim == 4:
            grid.point_data[key] = _vtk_flat_vector(arr)

    grid.save(path)


def _write_dvc_vtk(dvc: DVC, path: str, pv) -> None:
    nx, ny, nz = dvc.shape
    n = nx * ny * nz

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing    = tuple(dvc.spacing)
    grid.origin     = tuple(dvc.origin)

    grid.point_data["displacement"] = _vtk_flat_vector(dvc.displacement)

    for key, arr in dvc.strain.items():
        if arr.ndim == 3:
            grid.point_data[key] = _vtk_flat_scalar(arr)
        elif arr.ndim == 4:
            grid.point_data[key] = _vtk_flat_vector(arr)

    grid.save(path)


# ---------------------------------------------------------------------------
# DREAM.3D export
# ---------------------------------------------------------------------------

def write_dream3d(
    path: str,
    dct: DCT,
    euler_degrees: bool = True,
    voxel_size: float | None = None,
    phase_id: int = 1,
) -> None:
    """Write a DREAM.3D 6.x compatible HDF5 file.

    The output can be opened directly in DREAM.3D or ParaView (with the
    DREAM.3D plugin) for multimodal visualisation alongside DVC results.

    Required DCT attributes
    -----------------------
    ``GIDvol``      grain ID volume (1-based, 0 = unindexed).
    ``EulerAngle``  per-grain Euler angles, shape ``(n_grains, 3)``, Bunge ZXZ.
    ``IPF001``      per-voxel IPF RGB, shape ``(nx, ny, nz, 3)``.
    ``Mask``        binary sample mask, shape ``(nx, ny, nz)``.

    Parameters
    ----------
    path:
        Output file path (``*.dream3d``).
    dct:
        DCT object to export.
    euler_degrees:
        Whether ``dct.EulerAngle`` is stored in degrees.  DREAM.3D expects
        radians; conversion is applied automatically.
    voxel_size:
        Isotropic voxel size in µm.  Defaults to ``dct.voxel_size``.
    phase_id:
        Phase index written to the ``Phases`` CellData array (default 1).
    """
    import h5py

    nx, ny, nz = dct.shape
    n_cells     = nx * ny * nz
    vs          = voxel_size if voxel_size is not None else dct.voxel_size
    tuple_dims  = [nx, ny, nz]                    # DREAM.3D (x, y, z)

    # Per-grain Euler angles in radians (1-based: grain gid → row gid-1)
    euler = np.asarray(getattr(dct, "EulerAngle", np.zeros((1, 3))), dtype=np.float32)
    if euler_degrees:
        euler = np.deg2rad(euler)

    gid_flat  = dct.GIDvol.ravel(order="F").astype(np.int32)   # x-fastest
    n_grains  = int(gid_flat.max())                             # max grain ID

    # Per-voxel Euler angles: grain gid (1-based) → euler row gid-1
    euler_padded = np.zeros((n_grains + 1, 3), dtype=np.float32)
    n_rows = min(len(euler), n_grains)
    euler_padded[1 : n_rows + 1] = euler[:n_rows]              # row 0 = unindexed → 0
    euler_per_voxel = euler_padded[np.clip(gid_flat, 0, n_grains)]  # (N, 3)

    # Per-voxel phases
    phases_voxel = np.where(gid_flat > 0, phase_id, 0).astype(np.int32).reshape(-1, 1)

    with h5py.File(path, "w") as hout:
        # ── root attribute ──────────────────────────────────────────────
        hout.attrs["FileVersion"] = np.bytes_("7.0")

        # ── DataContainerBundles (required but empty) ────────────────────
        hout.require_group("DataContainerBundles")

        # ── DataContainer ────────────────────────────────────────────────
        dc = hout.require_group("DataContainers/ImageGeomDataContainer")

        # ── Geometry ─────────────────────────────────────────────────────
        geo = dc.require_group("_SIMPL_GEOMETRY")
        geo.create_dataset("GEOMETRY_TYPE",           data=np.bytes_("ImageGeometry"))
        geo.create_dataset("GEOMETRY_TYPE_ID",        data=np.uint32(0))
        geo.create_dataset("SPATIAL_DIMENSION_COUNT", data=np.uint32(3))
        geo.create_dataset("DIMENSIONS",  data=np.array([nx, ny, nz], dtype=np.uint64))
        geo.create_dataset("ORIGIN",      data=np.array([0., 0., 0.], dtype=np.float32))
        geo.create_dataset("SPACING",     data=np.array([vs, vs, vs], dtype=np.float32))

        # ── CellData ─────────────────────────────────────────────────────
        cell = dc.require_group("CellData")
        cell.attrs["AttributeMatrixType"] = np.uint32(3)        # Cell
        cell.attrs["TupleDimensions"]     = np.array(tuple_dims, dtype=np.uint64)

        def _add_cell(name, data, comp_dims):
            ds = cell.create_dataset(name, data=data, compression="gzip",
                                     compression_opts=4)
            _set_dream3d_attrs(ds, comp_dims, _dream3d_type(data), tuple_dims)

        _add_cell("FeatureIds",  gid_flat.reshape(-1, 1),          [1])
        _add_cell("Phases",      phases_voxel,                     [1])
        _add_cell("EulerAngles", euler_per_voxel,                  [3])
        _add_cell("Mask",
                  _dream3d_flat_scalar(dct.Mask.astype(np.uint8)), [1])

        if hasattr(dct, "IPF001"):
            ipf = dct.IPF001
            if ipf.dtype != np.uint8:
                ipf = (np.clip(ipf, 0.0, 1.0) * 255).astype(np.uint8)
            _add_cell("IPFColors", _dream3d_flat_vector(ipf),      [3])

        # Any additional 3-D volumes from the DCT (GIDvol already covered)
        skip = {"GIDvol", "Mask", "IPF001", "EulerAngle", "VoxSize",
                "RodVec", "Coord", "Diameter", "Dimension"}
        for key in dct._vol_keys:
            if key in skip:
                continue
            arr = getattr(dct, key)
            if arr.ndim == 3:
                flat = _dream3d_flat_scalar(arr.astype(np.float32))
                _add_cell(key, flat, [1])
            elif arr.ndim == 4:
                flat = _dream3d_flat_vector(arr.astype(np.float32))
                _add_cell(key, flat, [arr.shape[3]])

        # ── CellFeatureData ──────────────────────────────────────────────
        feat = dc.require_group("CellFeatureData")
        feat.attrs["AttributeMatrixType"] = np.uint32(4)          # CellFeature
        feat.attrs["TupleDimensions"]     = np.array([n_grains + 1], dtype=np.uint64)

        def _add_feat(name, data, comp_dims):
            ds = feat.create_dataset(name, data=data, compression="gzip",
                                     compression_opts=4)
            _set_dream3d_attrs(ds, comp_dims, _dream3d_type(data),
                               [n_grains + 1])

        active = np.zeros(n_grains + 1, dtype=np.uint8)
        active[1:] = 1
        _add_feat("Active",      active.reshape(-1, 1),            [1])
        _add_feat("EulerAngles", euler_padded,                     [3])

        feat_phases = np.zeros(n_grains + 1, dtype=np.int32)
        feat_phases[1:] = phase_id
        _add_feat("Phases",      feat_phases.reshape(-1, 1),       [1])

        # ── Pipeline (minimal — required by DREAM.3D reader) ─────────────
        pipe = hout.require_group("Pipeline")
        pipe.attrs["Number_Filters"] = np.int32(1)
        step = pipe.require_group("0")
        step.create_dataset(
            "Filter_Human_Label",
            data=np.bytes_("Exported by multimodal-registration"))
        step.create_dataset(
            "Filter_Name",
            data=np.bytes_("DataContainerWriter"))


# ---------------------------------------------------------------------------
# Format dispatcher
# ---------------------------------------------------------------------------

_EXT_MAP = {
    ".h5":       "h5",
    ".hdf5":     "h5",
    ".vtk":      "vtk",
    ".vts":      "vtk",
    ".vtu":      "vtk",
    ".dream3d":  "dream3d",
}


def write(
    path: str,
    dct: DCT | None = None,
    pct: ReferencePCT | None = None,
    dvc: DVC | None = None,
    **kwargs,
) -> None:
    """Write data to *path*, inferring the format from the file extension.

    Supported extensions: ``.h5``, ``.hdf5``, ``.vtk``, ``.vts``, ``.vtu``,
    ``.dream3d``.

    Parameters
    ----------
    path:
        Output file path.
    dct:
        DCT object.
    pct:
        PCT object (HDF5 output only).
    dvc:
        DVC object (HDF5 and VTK outputs only).
    **kwargs:
        Passed through to the format-specific writer.
    """
    ext = Path(path).suffix.lower()
    fmt = _EXT_MAP.get(ext)

    if fmt is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. "
            f"Supported: {list(_EXT_MAP)}"
        )

    if fmt == "h5":
        write_h5(path, dct=dct, pct=pct, dvc=dvc, **kwargs)
    elif fmt == "vtk":
        write_vtk(path, dct=dct, dvc=dvc, **kwargs)
    elif fmt == "dream3d":
        if dct is None:
            raise ValueError("write_dream3d requires a DCT object.")
        write_dream3d(path, dct, **kwargs)

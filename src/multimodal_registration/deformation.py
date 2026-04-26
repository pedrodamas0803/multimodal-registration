"""
Apply DVC displacement fields to DCT volumes and update crystal orientations.

Pipeline
--------
1. :func:`compute_deformation_gradient` — F = I + ∇U on the DVC grid.
2. :func:`extract_rotation_field`       — polar-decompose F → R via batched SVD.
3. :func:`warp_dct`                     — pull-back warp all DCT volumes into the
                                          deformed configuration.
4. :func:`update_ipf`                   — rotate per-grain orientations and
                                          recompute IPF colours (requires orix).
5. :func:`apply_dvc`                    — convenience wrapper for 3 + 4.

Coordinate conventions
-----------------------
* DCT and PCT share the same voxel grid after registration.
* DVC covers a sub-region (ROI) of that grid; its position is encoded in
  ``dvc.origin`` and ``dvc.spacing``.
* All spatial coordinates are in **PCT voxel units**.

Crystallographic conventions
------------------------------
* Euler angles are Bunge (ZXZ) convention.
* The orientation matrix **g** maps crystal → sample: ``d_sample = g @ d_crystal``.
* Local lattice rotation:  **g_new = R @ g_old**.

Requirements
------------
IPF colour computation requires **orix**::

    pip install orix
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import map_coordinates

from .dct import DCT
from .dvc import DVC


# ---------------------------------------------------------------------------
# Step 1 — Deformation gradient
# ---------------------------------------------------------------------------

def compute_deformation_gradient(dvc: DVC) -> np.ndarray:
    """Compute the Lagrangian deformation gradient field **F = I + ∇U**.

    Parameters
    ----------
    dvc:
        DVC object whose ``displacement`` array has shape ``(nx, ny, nz, 3)``.
        Spatial derivatives are computed using the voxel spacings in
        ``dvc.spacing``.

    Returns
    -------
    F : ndarray, shape ``(nx, ny, nz, 3, 3)``
        ``F[..., i, j]`` = δᵢⱼ + ∂Uᵢ/∂xⱼ
    """
    U  = dvc.displacement                       # (nx, ny, nz, 3)
    sp = dvc.spacing                            # (dx, dy, dz)
    shape = U.shape[:3]

    F = np.zeros(shape + (3, 3), dtype=float)
    for i in range(3):
        F[..., i, i] = 1.0                      # identity diagonal

    for i in range(3):                          # displacement component Uᵢ
        grads = np.gradient(U[..., i], sp[0], sp[1], sp[2])
        for j in range(3):                      # derivative direction xⱼ
            F[..., i, j] += grads[j]

    return F


# ---------------------------------------------------------------------------
# Step 2 — Rotation field via SVD polar decomposition
# ---------------------------------------------------------------------------

def extract_rotation_field(F: np.ndarray) -> np.ndarray:
    """Extract the rotation part **R** of **F** via polar decomposition.

    Uses the SVD factorisation: F = U S Vᵀ  →  R = U Vᵀ ∈ SO(3).
    Vectorised over all leading (batch) dimensions.

    Parameters
    ----------
    F : ndarray, shape ``(..., 3, 3)``

    Returns
    -------
    R : ndarray, shape ``(..., 3, 3)``
        Proper rotation matrices (det R = +1).
    """
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    # Ensure proper rotations — det = −1 can appear near singular F
    det  = np.linalg.det(R)
    fix  = np.where(det < 0, -1.0, 1.0)
    U[..., 2] *= fix[..., np.newaxis]           # flip last column of U
    return U @ Vt


# ---------------------------------------------------------------------------
# Internal coordinate helpers
# ---------------------------------------------------------------------------

def _dct_coords_in_dvc_grid(dct_shape: tuple, dvc: DVC) -> np.ndarray:
    """Fractional DVC-grid coordinates for every DCT voxel.

    Returns
    -------
    coords : ndarray, shape ``(3, nx*ny*nz)``
        Values outside [0, N−1] lie outside the DVC ROI.
    """
    nx, ny, nz = dct_shape
    g0, g1, g2 = np.mgrid[0:nx, 0:ny, 0:nz]
    pct = np.stack([g0, g1, g2], axis=0).reshape(3, -1).astype(float)

    origin  = np.array(dvc.origin,  dtype=float)[:, None]
    spacing = np.array(dvc.spacing, dtype=float)[:, None]
    return (pct - origin) / spacing             # (3, N)


def _interpolate_field(
    field: np.ndarray,
    coords: np.ndarray,
    cval: float = 0.0,
) -> np.ndarray:
    """Interpolate *field* defined on a DVC grid at fractional *coords*.

    Parameters
    ----------
    field : ndarray, shape ``(nx, ny, nz, ...)``
    coords : ndarray, shape ``(3, N)``
    cval : fill value for points outside the grid

    Returns
    -------
    out : ndarray, shape ``(N, ...)``
    """
    spatial = field.shape[:3]
    trailing = field.shape[3:]
    n = coords.shape[1]

    if trailing:
        n_comp = int(np.prod(trailing))
        flat   = field.reshape(spatial + (n_comp,))
        out    = np.full((n, n_comp), cval, dtype=float)
        for c in range(n_comp):
            out[:, c] = map_coordinates(
                flat[..., c], coords, order=1, mode="constant", cval=cval)
        return out.reshape((n,) + trailing)
    else:
        out = np.empty(n, dtype=float)
        out[:] = map_coordinates(
            field, coords, order=1, mode="constant", cval=cval)
        return out


def _reorthogonalise(R: np.ndarray) -> np.ndarray:
    """Project ``(N, 3, 3)`` matrices back to SO(3) after interpolation."""
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    det = np.linalg.det(Ro)
    fix = np.where(det < 0, -1.0, 1.0)
    U[..., 2] *= fix[:, None]
    return U @ Vt


def _outside_roi_mask(coords: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """Boolean mask: True where coords fall outside the DVC grid."""
    outside = np.zeros(coords.shape[1], dtype=bool)
    for ax, size in enumerate(grid_shape):
        outside |= (coords[ax] < 0) | (coords[ax] > size - 1)
    return outside


# ---------------------------------------------------------------------------
# Step 3 — Warp DCT volumes
# ---------------------------------------------------------------------------

def warp_dct(dct: DCT, dvc: DVC) -> None:
    """Warp all DCT volumes in-place using the DVC displacement field.

    Uses a pull-back (inverse) mapping::

        src(x) = x − U(x)
        warped[x] = dct[src(x)]

    Voxels outside the DVC ROI carry zero displacement (identity warp).
    Integer arrays (grain IDs, masks) use nearest-neighbour interpolation;
    float/RGB arrays use linear interpolation.

    Parameters
    ----------
    dct:
        DCT object (upscaled, padded and registered to PCT frame).
    dvc:
        DVC object.
    """
    nx, ny, nz = dct.shape
    dvc_coords = _dct_coords_in_dvc_grid(dct.shape, dvc)   # (3, N)

    # Displacement at every DCT voxel (PCT voxel units); zero outside ROI
    U_dct = _interpolate_field(dvc.displacement, dvc_coords, cval=0.0)  # (N, 3)

    pct_coords = np.array(np.mgrid[0:nx, 0:ny, 0:nz], dtype=float).reshape(3, -1)
    src_coords = (pct_coords - U_dct.T).reshape(3, nx, ny, nz)

    for key in dct._vol_keys:
        arr: np.ndarray = getattr(dct, key)
        order = 0 if np.issubdtype(arr.dtype, np.integer) else 1

        if arr.ndim == 3:
            setattr(dct, key,
                    map_coordinates(arr, src_coords, order=order, mode="nearest"))
        elif arr.ndim == 4:
            warped = np.empty_like(arr)
            for c in range(arr.shape[3]):
                warped[..., c] = map_coordinates(
                    arr[..., c], src_coords, order=1, mode="nearest")
            setattr(dct, key, warped)


# ---------------------------------------------------------------------------
# Step 4 — Update IPF map
# ---------------------------------------------------------------------------

_SYMMETRY_ALIASES: dict[str, str] = {
    # common shorthand → orix Hermann-Mauguin / Schoenflies strings
    "cubic":       "m-3m",
    "hexagonal":   "6/mmm",
    "tetragonal":  "4/mmm",
    "orthorhombic": "mmm",
    "trigonal":    "-3m",
    "monoclinic":  "2/m",
    "triclinic":   "-1",
}


def update_ipf(
    dct: DCT,
    dvc: DVC,
    crystal_symmetry: str = "cubic",
    reference_direction: str = "001",
    euler_degrees: bool = True,
) -> None:
    """Recompute the IPF colour map after applying DVC lattice rotations.

    For every DCT voxel:

    1. Retrieve the grain orientation from
       ``dct.EulerAngle[dct.GIDvol[i, j, k]]``.
    2. Rotate it by the local DVC rotation **R** (polar decomposition of
       the deformation gradient interpolated to the DCT grid).
    3. Recompute the IPF colour using *orix* under the given crystal symmetry.

    The result is written back to ``dct.IPF001``.

    Parameters
    ----------
    dct:
        DCT object.  Must expose ``EulerAngle`` (shape ``(n_grains, 3)``,
        Bunge ZXZ) and ``GIDvol`` (integer grain-ID volume).
    dvc:
        DVC object.
    crystal_symmetry:
        Crystal point-group symmetry.  Accepts common aliases
        (``'cubic'``, ``'hexagonal'``, …) or any Hermann-Mauguin /
        Schoenflies string understood by orix (e.g. ``'m-3m'``).
    reference_direction:
        Sample axis for the IPF projection: ``'001'`` (default),
        ``'010'``, or ``'100'``.
    euler_degrees:
        ``True`` if ``dct.EulerAngle`` is stored in degrees (typical for
        LabDCT / GrainMapper3D output).
    """
    # --- orix imports (version-tolerant) -----------------------------------
    try:
        from orix.quaternion import Rotation
        from orix.vector import Vector3d
        from orix.quaternion.symmetry import get_point_group
    except ImportError as e:
        raise ImportError(
            "orix is required for IPF colour computation.\n"
            "Install it with:  pip install orix"
        ) from e

    try:
        from orix.plot import IPFColorKey
    except ImportError:
        try:
            from orix.plot.colors import IPFColorKey
        except ImportError as e:
            raise ImportError(
                "Could not import IPFColorKey from orix.  "
                "Please update orix: pip install --upgrade orix"
            ) from e

    # --- validate DCT attributes -------------------------------------------
    for attr in ("EulerAngle", "GIDvol"):
        if not hasattr(dct, attr):
            raise AttributeError(
                f"dct.{attr} not found.  Make sure '{attr}' is stored under "
                f"the 'DS' group in the DCT HDF5 file."
            )

    # --- Step A: per-grain orientations ------------------------------------
    euler = np.asarray(dct.EulerAngle, dtype=float)   # (n_grains, 3)
    g_per_grain = Rotation.from_euler(
        euler, degrees=euler_degrees)                   # (n_grains,)

    # --- Step B: rotation field on DCT grid --------------------------------
    F     = compute_deformation_gradient(dvc)          # (nx_d, ny_d, nz_d, 3, 3)
    R_dvc = extract_rotation_field(F)                  # (nx_d, ny_d, nz_d, 3, 3)

    dvc_coords = _dct_coords_in_dvc_grid(dct.shape, dvc)       # (3, N)
    outside    = _outside_roi_mask(dvc_coords, R_dvc.shape[:3])

    # Interpolate 9 matrix components; fill outside with 0 → identity below
    R_interp = _interpolate_field(R_dvc, dvc_coords, cval=0.0)   # (N, 3, 3)
    R_interp[outside] = np.eye(3)                     # identity outside ROI
    R_interp = _reorthogonalise(R_interp)              # re-project to SO(3)

    R_voxel = Rotation.from_matrix(R_interp)           # (N,)

    # --- Step C: per-voxel grain orientations ------------------------------
    # MATLAB reconstruction codes use 1-based grain IDs (0 = unindexed).
    # EulerAngle row 0 → grain 1, row 1 → grain 2, etc.
    gid_flat   = dct.GIDvol.ravel().astype(int)        # (N,)  values 0..n_grains
    indexed    = gid_flat > 0                          # mask of indexed voxels
    row_idx    = np.clip(gid_flat - 1, 0, len(g_per_grain) - 1)  # 0-based rows

    # EulerAngle shape: h5py reads MATLAB (3, n_grains) as (n_grains, 3) — correct.
    # But if it comes out (3, n_grains), transpose it.
    if euler.shape[0] == 3 and euler.ndim == 2:
        euler = euler.T
        g_per_grain = Rotation.from_euler(euler, degrees=euler_degrees)

    g_voxel = g_per_grain[row_idx]                     # (N,)

    # --- Step D: apply local rotation  g_new = R · g_old ------------------
    g_new = R_voxel * g_voxel                          # (N,)

    # --- Step E: IPF colours -----------------------------------------------
    sym_str = _SYMMETRY_ALIASES.get(crystal_symmetry.lower(), crystal_symmetry)
    pg = get_point_group(sym_str)

    dir_map = {
        "001": Vector3d.zvector(),
        "010": Vector3d.yvector(),
        "100": Vector3d.xvector(),
    }
    ref_dir = dir_map.get(reference_direction, Vector3d.zvector())

    ipf_key = IPFColorKey(pg, direction=ref_dir)
    rgb = np.asarray(ipf_key.orientation2color(g_new), dtype=float)  # (N, 3) in [0,1]

    # Unindexed voxels (GID == 0) → black
    rgb[~indexed] = 0.0

    nx, ny, nz = dct.shape

    # Preserve original dtype (uint8 0–255 vs float 0–1)
    if dct.IPF001.dtype == np.uint8:
        dct.IPF001 = (rgb * 255).astype(np.uint8).reshape(nx, ny, nz, 3)
    else:
        dct.IPF001 = rgb.astype(dct.IPF001.dtype).reshape(nx, ny, nz, 3)


# ---------------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------------

def apply_dvc(
    dct: DCT,
    dvc: DVC,
    crystal_symmetry: str = "cubic",
    reference_direction: str = "001",
    euler_degrees: bool = True,
    update_orientations: bool = True,
) -> None:
    """Apply DVC results to DCT in-place: warp volumes then update IPF.

    Calls :func:`warp_dct` followed by :func:`update_ipf`.

    Parameters
    ----------
    dct:
        DCT object (upscaled, padded, registered to PCT frame).
    dvc:
        DVC object.
    crystal_symmetry:
        Point-group symmetry string (default ``'cubic'``).
    reference_direction:
        IPF projection axis (``'001'``, ``'010'``, or ``'100'``).
    euler_degrees:
        Whether ``dct.EulerAngle`` is stored in degrees.
    update_orientations:
        If ``False``, skip the IPF update and only warp the spatial volumes.
        Useful for a quick geometric check without requiring orix.
    """
    warp_dct(dct, dvc)

    if update_orientations:
        update_ipf(
            dct, dvc,
            crystal_symmetry=crystal_symmetry,
            reference_direction=reference_direction,
            euler_degrees=euler_degrees,
        )

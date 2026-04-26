# Deformation & Rotations

DVC measures how a 3-D volume deforms between two loading states.  This page
explains how the package converts a DVC displacement field into a local
rotation field and uses it to warp DCT volumes and update crystal orientations.

---

## The displacement field

DVC produces a displacement field \(\mathbf{U}(\mathbf{X})\) defined on a
regular grid in the **reference** (undeformed) configuration.  A material
point at position \(\mathbf{X}\) in the reference moves to:

\[
\mathbf{x} = \mathbf{X} + \mathbf{U}(\mathbf{X})
\]

in the deformed configuration.  The field \(\mathbf{U}\) is typically stored
as a 4-D array of shape \((N_x, N_y, N_z, 3)\).

---

## Deformation gradient

The local deformation is fully described by the **deformation gradient tensor**:

\[
\mathbf{F} = \mathbf{I} + \nabla \mathbf{U}, \qquad
F_{ij} = \delta_{ij} + \frac{\partial U_i}{\partial X_j}
\]

\(\mathbf{F}\) is a \(3 \times 3\) tensor field computed numerically using
central finite differences (`numpy.gradient`) with the DVC grid spacing as
the step size.

!!! note "Spatial derivatives"
    The gradient is taken with respect to the **reference** coordinates
    \(\mathbf{X}\), consistent with the Lagrangian description used by DVC.

---

## Polar decomposition

Any invertible tensor \(\mathbf{F}\) (with \(\det \mathbf{F} > 0\)) admits a
unique right polar decomposition:

\[
\mathbf{F} = \mathbf{R}\,\mathbf{U}_s
\]

where \(\mathbf{R} \in \mathrm{SO}(3)\) is a proper rotation matrix and
\(\mathbf{U}_s\) is a symmetric positive-definite stretch tensor.

### SVD route

The rotation \(\mathbf{R}\) is extracted efficiently via the singular value
decomposition:

\[
\mathbf{F} = \mathbf{U}\,\mathbf{S}\,\mathbf{V}^\top
\implies
\mathbf{R} = \mathbf{U}\,\mathbf{V}^\top
\]

This is evaluated **in batch** over all voxels using `numpy.linalg.svd`,
which vectorises over arbitrary leading dimensions:

```python
U, S, Vt = np.linalg.svd(F)          # F shape: (..., 3, 3)
R = U @ Vt                            # R shape: (..., 3, 3)
```

A sign correction ensures \(\det \mathbf{R} = +1\) (proper rotation) even
near singular points where numerical noise can introduce a reflection.

---

## Warping DCT volumes

To map DCT data from the reference into the deformed configuration, a
**pull-back** (inverse) warp is used.  For each output voxel position
\(\mathbf{x}\), the source position in the reference is approximated as:

\[
\mathbf{X}(\mathbf{x}) \approx \mathbf{x} - \mathbf{U}(\mathbf{x})
\]

This is exact for small displacements and a good approximation for typical DVC
fields.  The warped volume is then:

\[
f_\text{deformed}(\mathbf{x}) = f_\text{ref}\!\left(\mathbf{x} - \mathbf{U}(\mathbf{x})\right)
\]

implemented via `scipy.ndimage.map_coordinates` with order-0 (nearest-neighbour)
for integer arrays and order-1 (trilinear) for float arrays.

### Coordinate mapping

The DVC field covers only a **ROI** within the PCT frame.  Voxels outside the
ROI receive zero displacement (identity warp), localised using `dvc.origin`
and `dvc.spacing`:

\[
\mathbf{p}_\text{DVC} = \frac{\mathbf{p}_\text{PCT} - \mathbf{o}}{\Delta}
\]

where \(\mathbf{o}\) is the ROI origin and \(\Delta\) is the DVC grid spacing,
both in PCT voxel units.

---

## Rotating grain orientations

The rotation field \(\mathbf{R}(\mathbf{X})\) describes how each material
element has rotated from the reference to the deformed state.  The crystal
orientation at a voxel transforms as:

\[
\mathbf{g}_\text{new} = \mathbf{R}\,\mathbf{g}_\text{old}
\]

where \(\mathbf{g}\) is the orientation matrix mapping crystal axes to sample
axes (see [Crystal Orientations & IPF](crystallography.md)).

### Interpolation to the DCT grid

\(\mathbf{R}\) is defined on the (coarser) DVC grid and must be interpolated
to the (finer) DCT grid.  The nine matrix components are interpolated
independently with trilinear interpolation, after which each matrix is
**re-projected to \(\mathrm{SO}(3)\)**  via a second SVD step to restore
orthogonality lost during interpolation:

\[
\tilde{\mathbf{R}} = \text{interp}(\mathbf{R}_\text{DVC}) \xrightarrow{\text{SVD}} \hat{\mathbf{R}} \in \mathrm{SO}(3)
\]

Voxels outside the DVC ROI are assigned the identity rotation
(\(\mathbf{R} = \mathbf{I}\)), meaning no lattice rotation is applied there.

---

## References

- Sutton M A *et al.* (2009). *Image Correlation for Shape, Motion and
  Deformation Measurements.* Springer.
- Bonet J, Wood R D (2008). *Nonlinear Continuum Mechanics for Finite Element
  Analysis*, 2nd ed. Cambridge University Press. (§3 — kinematics)
- Guizar-Sicairos M *et al.* (2011). *Phase tomography from x-ray coherent
  diffractive imaging.* Physical Review Letters **107**, 022607.

# Crystal Orientations & IPF

## Orientation representation

A crystal orientation describes how the crystal lattice frame is rotated
relative to the sample (laboratory) frame.  The package follows the convention
used by LabDCT / GrainMapper3D:

- **Bunge Euler angles** \((\varphi_1,\, \Phi,\, \varphi_2)\) — three
  successive rotations about the ZXZ axes of the sample frame.
- Stored in `dct.EulerAngle` with shape \((N_\text{grains}, 3)\) in degrees.
- Grain `gid` (1-based, MATLAB convention) maps to row `gid − 1`.

### From Euler angles to rotation matrix

The Bunge rotation matrix \(\mathbf{g}\) is the composition:

\[
\mathbf{g} = \mathbf{R}_Z(\varphi_2)\,\mathbf{R}_X(\Phi)\,\mathbf{R}_Z(\varphi_1)
\]

It maps a direction \(\mathbf{d}\) expressed in the **crystal** frame to the
corresponding direction in the **sample** frame:

\[
\mathbf{d}_\text{sample} = \mathbf{g}\,\mathbf{d}_\text{crystal}
\]

The inverse (transposed, since \(\mathbf{g} \in \mathrm{SO}(3)\)) gives the
crystal direction that corresponds to a given sample direction:

\[
\mathbf{d}_\text{crystal} = \mathbf{g}^\top \mathbf{d}_\text{sample}
\]

---

## Inverse Pole Figure (IPF)

An Inverse Pole Figure maps a chosen **sample reference direction**
(e.g. the loading axis \([001]\)) into the **crystal** frame and assigns an
RGB colour to the result.

### Step 1 — project into crystal frame

For the \([001]\) IPF:

\[
\mathbf{d} = \mathbf{g}^\top \begin{pmatrix}0\\0\\1\end{pmatrix}
\]

### Step 2 — apply crystal symmetry

Under cubic (m-3m) symmetry there are 48 equivalent directions.  The
representative in the **standard triangle** (fundamental zone) is obtained by:

1. Taking absolute values: \(\mathbf{d} \leftarrow |\mathbf{d}|\).
2. Sorting the components in descending order: \(d_0 \geq d_1 \geq d_2 \geq 0\).

The three corners of the standard triangle and their conventional colours are:

| Direction | Indices | Colour |
|-----------|---------|--------|
| \([001]\) | \((1,0,0)\) after reduction | Blue |
| \([101]\) | \((1,1,0)/\sqrt{2}\) | Green |
| \([111]\) | \((1,1,1)/\sqrt{3}\) | Red |

!!! warning "Colour convention"
    The exact colour mapping depends on the reconstruction software.
    The pre-computed `IPF001` stored in the DCT HDF5 reflects the convention
    of the MATLAB reconstruction code.  After a DVC rotation update the
    colours are recomputed by **orix**, which follows the MTEX convention.
    Verify visually that the two conventions match before comparing maps.

### Step 3 — colour assignment

orix computes IPF colours using a stereographic projection of the fundamental
triangle, normalising so that each corner maps to pure red, green, or blue
and all intermediate orientations are interpolated linearly in the triangle.

---

## Updating orientations after DVC

When a DVC rotation \(\mathbf{R}\) is applied to a grain with orientation
\(\mathbf{g}\), the new orientation is:

\[
\mathbf{g}_\text{new} = \mathbf{R}\,\mathbf{g}_\text{old}
\]

The crystal direction now aligned with the sample \([001]\) axis becomes:

\[
\mathbf{d}_\text{new} = \mathbf{g}_\text{new}^\top \begin{pmatrix}0\\0\\1\end{pmatrix}
= \mathbf{g}_\text{old}^\top \mathbf{R}^\top \begin{pmatrix}0\\0\\1\end{pmatrix}
\]

Physically this means: the lattice rotation \(\mathbf{R}\) tilts the crystal
axes, so the sample \([001]\) direction now points along a different crystal
direction — the IPF colour changes accordingly.

Because \(\mathbf{R}\) varies **spatially** within the DVC field, different
regions of the same grain can show slightly different colours in the updated
IPF map, reflecting **sub-grain orientation gradients** introduced by the
deformation.

---

## Rodrigues vectors

Some DCT codes also store orientations as **Rodrigues–Frank vectors**
(`dct.RodVec`), related to the rotation axis \(\hat{\mathbf{n}}\) and angle
\(\omega\) by:

\[
\mathbf{r} = \hat{\mathbf{n}}\tan\!\left(\frac{\omega}{2}\right)
\]

orix can convert these directly via `Rotation.from_axes_angles` after
normalising, or via `Rotation.from_rodrigues` in older versions.

---

## References

- Bunge H J (1982). *Texture Analysis in Materials Science.* Butterworth.
- Nolze G, Hielscher R (2016). *Orientations — perfectly colored.*
  Journal of Applied Crystallography **49**, 1786–1802.
- Bachmann F, Hielscher R, Schaeben H (2010). *Texture analysis with MTEX.*
  Solid State Phenomena **160**, 63–68.
- orix documentation: [orix.readthedocs.io](https://orix.readthedocs.io)

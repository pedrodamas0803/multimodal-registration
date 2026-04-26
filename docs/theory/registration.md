# Image Registration

Registration is the process of finding a geometric transformation that brings
two volumes into the same frame of reference.  Here we deal exclusively with
**rigid translation** — the same physical sample is imaged by two different
techniques, so only a shift (and possibly a flip) is needed to align them.

---

## Why the modalities need aligning

DCT and PCT are acquired on the same sample but in separate experiments, often
on different beamlines.  The sample may be:

- **Mounted differently** between experiments (translation, rotation).
- **Scanned at a different voxel size** — DCT voxel sizes are typically
  5–20 µm, while PCT can reach 0.5–2 µm.
- **Stored with a different spatial origin** in the reconstruction software.

The registration pipeline corrects for all of these.

---

## Upscaling

Before any alignment can be attempted, both volumes must share the same voxel
size.  A zoom factor is computed from the ratio of voxel sizes:

\[
z = \frac{\Delta x_\text{DCT}}{\Delta x_\text{PCT}}
\]

and applied independently to each spatial axis using `scipy.ndimage.zoom`
(order-0 nearest-neighbour for integer arrays such as `GIDvol`; order-1
bilinear for float arrays).

---

## Phase cross-correlation

Once the volumes share a voxel size and have been padded to the same shape,
the translation is estimated using **phase cross-correlation** in the Fourier
domain.

Given a fixed image \(f\) and a moving image \(g\), define the
cross-power spectrum:

\[
C(\mathbf{u}) = \frac{F(\mathbf{u})\, \overline{G(\mathbf{u})}}
                     {\lvert F(\mathbf{u})\, \overline{G(\mathbf{u})} \rvert}
\]

where \(F\) and \(G\) are the Fourier transforms of \(f\) and \(g\), and
\(\mathbf{u}\) is the frequency vector.  The inverse Fourier transform of
\(C\) is an impulse at the true translation vector:

\[
\mathcal{F}^{-1}\{C\}(\mathbf{x}) = \delta(\mathbf{x} - \mathbf{t})
\]

The peak location gives \(\mathbf{t}\) to sub-voxel accuracy (controlled by
`upsample_factor`).

!!! note "Why normalise?"
    Dividing by the magnitude of the cross-power spectrum makes the estimator
    insensitive to the absolute intensity scales of the two modalities — 
    essential here since DCT produces a binary mask while PCT produces a
    grey-level absorption contrast image.

---

## Why use masks

Correlating the raw DCT grain-ID volume against the PCT grey-level volume
would fail: the two signals carry completely different physical information.
Instead, both volumes are reduced to **binary masks** that capture the sample
geometry — the one quantity both modalities share.

| Volume | Mask derivation |
|--------|-----------------|
| DCT | `Mask` dataset loaded directly from the HDF5 file |
| PCT | Otsu threshold on the grey-level volume; optional binary closing to fill internal voids |

The shift found from the mask correlation is then applied to **all** DCT
volumes (grain IDs, IPF colours, etc.).

---

## Sub-pixel accuracy

Phase cross-correlation can localise the peak to sub-voxel precision by
upsampling the cross-power spectrum in a small neighbourhood around the
integer-pixel peak (Guizar-Sicairos *et al.*, 2008).  The `upsample_factor`
parameter controls the precision: a value of 10 gives 0.1-voxel accuracy.

---

## References

- Guizar-Sicairos M, Thurman S T, Fienup J R (2008). *Efficient subpixel
  image registration algorithms.* Optics Letters **33**(2), 156–158.
- Lewis J P (1995). *Fast normalized cross-correlation.* Vision Interface **10**.

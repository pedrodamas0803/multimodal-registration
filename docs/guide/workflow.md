# Full Pipeline

This page walks through the complete registration and deformation workflow
from raw files to updated IPF maps.

---

## 1 — Load data

```python
from multimodal_registration import DCT, ReferencePCT, DVC

# DCT: HDF5 file containing GIDvol, IPF001, Mask, EulerAngle, ...
dct = DCT("scan.h5")
print(dct.shape)        # (nx, ny, nz) of the DCT volume
print(dct.voxel_size)   # µm

# PCT: TIFF stack or HDF5
pct = ReferencePCT("reference.tiff")
pct = ReferencePCT("reference.h5", h5_key="/entry/data/volume")

# Optional: binary closing to fill pores in the PCT mask
pct = ReferencePCT("reference.tiff", closing_radius=5)
print(pct.shape)

# DVC: single-step VTK
dvc = DVC("results.vts",
          displacement_key="displacement",
          strain_keys=["strain"])

# DVC: multi-step — one VTK file per loading step
dvc = DVC(["step_01.vts", "step_02.vts", "step_03.vts"],
          displacement_key="displacement",
          strain_keys=["strain"],
          step=-1)          # start at last step

# DVC: multi-step MATLAB — groups /step_1/U, /step_2/U, ...
dvc = DVC("results.mat",
          displacement_key="U",
          step_keys=["step_1", "step_2", "step_3"],
          origin=(120, 80, 50),
          spacing=(2.0, 2.0, 2.0))

# DVC: multi-step MATLAB — 5-D array U shape (n_steps, nx, ny, nz, 3)
dvc = DVC("results.mat",
          displacement_key="U",
          origin=(120, 80, 50),
          spacing=(2.0, 2.0, 2.0))

print(dvc.n_steps, dvc.step)   # total steps, active step index
```

!!! tip "Discover VTK field names"
    If you are unsure of the field names in a VTK file:
    ```python
    dvc = DVC("results.vts", displacement_key="some_field")
    print(dvc.available_fields())
    ```

---

## 2 — Bring DCT into the PCT frame

### 2a — Flip

DCT reconstruction codes often output volumes with a vertical flip relative
to PCT.  Apply axis-0 flip before any other operation:

```python
dct.flip(axis=0)
```

### 2b — Upscale

Zoom the DCT to match the PCT voxel size:

```python
pct_vox_size = 0.65   # µm — read from your PCT metadata
dct.upscale(pct_vox_size)
```

Integer arrays (`GIDvol`, `Mask`) use nearest-neighbour; float / RGB arrays
use bilinear interpolation automatically.

### 2c — Pad

Centre-pad the DCT to the PCT volume shape:

```python
dct.pad(pct.shape)
```

---

## 3 — Automatic registration

```python
from multimodal_registration import register

shift = register(dct, pct)
print(f"Applied shift: {shift} voxels")
```

Both sides use binary masks by default (`dct.Mask` and `pct.mask`).
Sub-pixel precision is controlled by `upsample_factor` (default 10):

```python
shift = register(dct, pct, upsample_factor=50)
```

---

## 4 — Visual check

```python
from multimodal_registration import overlay_check

overlay_check(dct, pct, plane="xz", alpha=0.5)
```

---

## 5 — Manual fine-tuning (if needed)

```python
from multimodal_registration import ManualRegistration

mr = ManualRegistration(dct, pct, step=1.0)
mr.show()      # keyboard / widget UI
mr.apply()     # write accumulated shift + flips to dct
```

See [Interactive Registration](interactive.md) for keyboard shortcuts.

---

## 6 — Select deformation step

For multi-step DVC datasets, choose which loading step to use before applying
the deformation to the DCT:

```python
print(dvc.n_steps)      # total number of steps
print(dvc.step)         # currently active step (0-based)

# Switch to a specific step — reloads displacement (and strain for VTK)
dvc.select_step(2)      # third step
dvc.select_step(-1)     # last step

# Iterate over all steps
for i in range(dvc.n_steps):
    dvc.select_step(i)
    apply_dvc(dct_copy, dvc)
    write(f"output_step{i:02d}.dream3d", dct=dct_copy)
```

!!! note "Incremental vs cumulative displacements"
    DVC software may store either **incremental** displacements (step *i*
    relative to step *i − 1*) or **cumulative** displacements (step *i*
    relative to the reference).  Check your DVC software's convention before
    applying fields to the DCT — the two cases require different composition
    logic.

---

## 7 — Apply DVC


```python
from multimodal_registration import apply_dvc

# Warp all DCT volumes + update IPF colours
apply_dvc(dct, dvc,
          crystal_symmetry="cubic",
          reference_direction="001",
          euler_degrees=True)
```

To warp only (no IPF update, no orix required):

```python
apply_dvc(dct, dvc, update_orientations=False)
```

---

## 7 — Inspect results

```python
import matplotlib.pyplot as plt

mid = dct.shape[1] // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(dct.IPF001[:, mid, :])
axes[0].set_title("Updated IPF001")
axes[1].imshow(dct.GIDvol[:, mid, :], cmap="tab20")
axes[1].set_title("Warped GIDvol")
plt.show()
```

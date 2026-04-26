# Exporting Data

After registration and deformation, results can be exported to three formats
for downstream visualisation and analysis.

## Format overview

| Format | Extension | Use case |
|--------|-----------|----------|
| HDF5 | `.h5` | Lossless archive of all arrays; re-loadable by the package |
| VTK | `.vts` | ParaView, DREAM.3D plugin, any VTK-compatible viewer |
| DREAM.3D | `.dream3d` | DREAM.3D GUI, DREAM.3D ParaView plugin |

---

## HDF5

Writes DCT, PCT and DVC data into a single file under logical groups:

```
/DCT/DS/<key>      — all arrays loaded from the original DCT HDF5
/PCT/vol           — grey-level volume
/PCT/mask          — Otsu binary mask
/DVC/displacement  — (nx, ny, nz, 3) displacement field
/DVC/strain/<key>  — strain fields (if present)
/DVC/origin        — ROI origin in PCT voxel space
/DVC/spacing       — DVC grid spacing
```

```python
from multimodal_registration import write_h5

write_h5("results.h5", dct=dct, pct=pct, dvc=dvc)

# Or use the dispatcher
from multimodal_registration import write
write("results.h5", dct=dct, pct=pct, dvc=dvc)
```

---

## VTK

Writes uniform-grid (ImageData) VTK files readable by ParaView and DREAM.3D.
When both DCT and DVC are supplied, two files are created automatically:
`results_dct.vts` and `results_dvc.vts`.

```python
from multimodal_registration import write_vtk

# Both — produces results_dct.vts and results_dvc.vts
write_vtk("results.vts", dct=dct, dvc=dvc)

# DCT only
write_vtk("dct.vts", dct=dct)

# DVC only
write_vtk("dvc.vts", dvc=dvc)
```

All volumetric arrays in the DCT object (`GIDvol`, `Mask`, `IPF001`, …)
are written as point data.  DVC displacement and strain fields are written
to the DVC file on their own grid (with the correct `origin` and `spacing`).

!!! note
    Requires pyvista: `pip install "multimodal-registration[vtk]"`

---

## DREAM.3D

Produces a DREAM.3D 6.x compatible `.dream3d` file (HDF5 with a strict
internal structure) that can be opened directly in the DREAM.3D application
or via the DREAM.3D ParaView plugin.

```python
from multimodal_registration import write_dream3d

write_dream3d("results.dream3d", dct=dct, euler_degrees=True)
```

### What is written

| DREAM.3D path | Source | Notes |
|---------------|--------|-------|
| `CellData/FeatureIds` | `dct.GIDvol` | Grain IDs, 1-based |
| `CellData/EulerAngles` | `dct.EulerAngle` + `dct.GIDvol` | Per-voxel, radians |
| `CellData/IPFColors` | `dct.IPF001` | uint8 RGB |
| `CellData/Mask` | `dct.Mask` | uint8 |
| `CellData/Phases` | constant `phase_id` | Default 1 |
| `CellFeatureData/EulerAngles` | `dct.EulerAngle` | Per-grain, radians |
| `CellFeatureData/Active` | derived | All grains active |
| `_SIMPL_GEOMETRY` | `dct.voxel_size` | ImageGeometry |

!!! warning "Euler angle convention"
    DREAM.3D expects Euler angles in **radians** (Bunge ZXZ).  The writer
    converts from degrees automatically when `euler_degrees=True` (default).

### Opening in ParaView

1. Open ParaView and load the `.dream3d` file via
   **File → Open** (the DREAM.3D reader is built into ParaView ≥ 5.10).
2. In the pipeline browser, apply the reader and select which arrays to
   display (`FeatureIds`, `IPFColors`, etc.).
3. To overlay DVC results, load the `_dvc.vts` file as a separate source
   and use a **Resample To Image** or **Probe Location** filter to compare
   on the same grid.

---

## Format dispatcher

`write()` infers the format from the file extension:

```python
from multimodal_registration import write

write("results.h5",      dct=dct, pct=pct, dvc=dvc)
write("results.vts",     dct=dct, dvc=dvc)
write("results.dream3d", dct=dct)
```

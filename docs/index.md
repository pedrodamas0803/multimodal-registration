# multimodal-registration

A Python package for registering and combining 3-D volumes from different
synchrotron imaging modalities — primarily **Diffraction Contrast Tomography
(DCT)** and **Phase Contrast Tomography (PCT)** — and for propagating
**Digital Volume Correlation (DVC)** displacement fields into the crystallographic
data.

---

## What it does

| Step | Description |
|------|-------------|
| **Load** | Read DCT (HDF5), PCT (TIFF / HDF5), and DVC (VTK / MAT) volumes |
| **Register** | Upscale and pad DCT to the PCT voxel grid, then align using phase cross-correlation |
| **Inspect** | Interactive overlay for manual fine-tuning (Jupyter or script) |
| **Deform** | Warp DCT volumes with DVC displacement fields |
| **Rotate** | Update IPF colour maps to reflect DVC lattice rotations |

---

## Quick start

```python
from multimodal_registration import DCT, ReferencePCT, DVC
from multimodal_registration import register, apply_dvc

# 1 — load
dct = DCT("scan.h5")
pct = ReferencePCT("reference.tiff")
dvc = DVC("results.vts", displacement_key="displacement",
          strain_keys=["strain"])

# 2 — bring DCT into PCT frame
dct.flip(axis=0)                   # match orientation
dct.upscale(pct_vox_size)          # match voxel size
dct.pad(pct.shape)                 # match spatial extent

# 3 — automatic registration
shift = register(dct, pct)

# 4 — apply DVC deformation + update IPF
apply_dvc(dct, dvc, crystal_symmetry="cubic")
```

---

## Package layout

```
multimodal_registration/
├── backends.py       GPU / CPU detection (CuPy → NumPy fallback)
├── dct.py            DCT class  — load, upscale, pad, shift, flip
├── pct.py            ReferencePCT class — load, Otsu mask, binary closing
├── dvc.py            DVC class — VTK and MATLAB/HDF5 loaders
├── registration.py   Phase cross-correlation registration
├── interactive.py    ManualRegistration — Jupyter widgets / keyboard UI
└── deformation.py    Warp + IPF update from DVC displacement fields
```

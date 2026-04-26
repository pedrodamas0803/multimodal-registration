# multimodal-registration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python package for registering and combining 3-D volumes from synchrotron
imaging experiments — primarily **Diffraction Contrast Tomography (DCT)** and
**Phase Contrast Tomography (PCT)** — and for propagating **Digital Volume
Correlation (DVC)** displacement fields into the crystallographic data.

---

## What it does

| Step | Description |
|------|-------------|
| **Load** | Read DCT (HDF5), PCT (TIFF / HDF5), and DVC (VTK / MATLAB) volumes |
| **Register** | Upscale and pad DCT to the PCT voxel grid; align with phase cross-correlation |
| **Inspect** | Interactive cross-section overlay for manual fine-tuning |
| **Deform** | Warp DCT volumes using DVC displacement fields |
| **Rotate** | Update IPF colour maps to reflect DVC lattice rotations |
| **Export** | Write results as HDF5, VTK, or DREAM.3D files |

---

## Installation

```bash
# Core
pip install multimodal-registration

# With optional extras
pip install "multimodal-registration[vtk,jupyter,crystal]"

# NVIDIA GPU acceleration
pip install "multimodal-registration[cuda12]"   # or cuda11
```

### Using uv

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install "multimodal-registration[vtk,jupyter,crystal]"
```

### Using conda

```bash
conda create -n multimodal-reg python=3.10
conda activate multimodal-reg
conda install -c conda-forge numpy scipy h5py scikit-image matplotlib
pip install "multimodal-registration[vtk,jupyter,crystal]"
```

| Extra | Adds | Required for |
|-------|------|--------------|
| `cuda11` / `cuda12` | `cupy`, `cucim` | GPU-accelerated zoom, shift, registration |
| `vtk` | `pyvista` | VTK file I/O (DVC results) |
| `jupyter` | `ipywidgets`, `ipympl` | Interactive overlay in notebooks |
| `crystal` | `orix` | Rotating grain orientations, recomputing IPF colours |

---

## Quick start

```python
from multimodal_registration import (
    DCT, ReferencePCT, DVC,
    register, apply_dvc, write,
    backend_name,
)

print(backend_name())   # 'cuda' or 'cpu'

# 1 — load
dct = DCT("scan.h5")
pct = ReferencePCT("reference.tiff", closing_radius=5)
dvc = DVC(["step_01.vts", "step_02.vts", "step_03.vts"],
          displacement_key="displacement",
          strain_keys=["strain"])

# 2 — bring DCT into the PCT frame
dct.flip(axis=0)
dct.upscale(target_vox_size=0.65)   # µm
dct.pad(pct.shape)

# 3 — automatic registration (phase cross-correlation on binary masks)
shift = register(dct, pct)

# 4 — manual fine-tuning if needed
from multimodal_registration import ManualRegistration
mr = ManualRegistration(dct, pct, step=1.0)
mr.show()    # keyboard / widget UI
mr.apply()

# 5 — select DVC step and apply deformation
dvc.select_step(-1)           # last loading step
apply_dvc(dct, dvc, crystal_symmetry="cubic")

# 6 — export
write("results.dream3d", dct=dct)
write("results.h5",      dct=dct, pct=pct, dvc=dvc)
write("results.vts",     dct=dct, dvc=dvc)
```

---

## Package layout

```
multimodal_registration/
├── backends.py       GPU / CPU detection (CuPy → NumPy/SciPy fallback)
├── dct.py            DCT  — load, upscale, pad, flip, shift
├── pct.py            ReferencePCT — load, Otsu mask, optional binary closing
├── dvc.py            DVC  — VTK / MATLAB loader, multi-step support
├── registration.py   Phase cross-correlation alignment
├── interactive.py    ManualRegistration — Jupyter widgets / keyboard UI
├── deformation.py    Warp volumes + update IPF from DVC displacement fields
└── io.py             Export to HDF5, VTK, DREAM.3D
```

---

## Multi-step DVC

DVC results with multiple loading steps are supported in three layouts:

```python
# One VTK file per step
dvc = DVC(["step_01.vts", "step_02.vts", "step_03.vts"],
          displacement_key="displacement")

# Single HDF5 with one group per step: /step_1/U, /step_2/U, ...
dvc = DVC("results.mat", displacement_key="U",
          step_keys=["step_1", "step_2", "step_3"],
          origin=(120, 80, 50), spacing=(2.0, 2.0, 2.0))

# Single HDF5 with 5-D array: shape (n_steps, nx, ny, nz, 3)
dvc = DVC("results.mat", displacement_key="U",
          origin=(120, 80, 50), spacing=(2.0, 2.0, 2.0))

print(dvc.n_steps, dvc.step)
dvc.select_step(1)     # switch step — reloads displacement
```

---

## GPU acceleration

The active backend is selected automatically at import time:

```
CUDA (CuPy)  →  CPU (NumPy / SciPy)
```

```python
from multimodal_registration import backend_name
print(backend_name())   # 'cuda' or 'cpu'
```

GPU-accelerated operations: `upscale`, `shift`, `pad` (binary closing),
and phase cross-correlation registration.

---

## Export formats

```python
from multimodal_registration import write

write("results.h5",      dct=dct, pct=pct, dvc=dvc)   # HDF5
write("results.vts",     dct=dct, dvc=dvc)             # VTK ImageData
write("results.dream3d", dct=dct)                      # DREAM.3D 6.x
```

DREAM.3D files include `FeatureIds`, per-voxel `EulerAngles` (radians),
`IPFColors`, `Phases`, and `Mask` in the correct SIMPL hierarchy, ready to
open in DREAM.3D or ParaView.

---

## Documentation

Full documentation including theory background, API reference, and user
guides can be built locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

Topics covered:
- Image registration — phase cross-correlation, mask-based alignment
- Deformation mechanics — deformation gradient, polar decomposition, pull-back warp
- Crystallography — Bunge Euler angles, IPF colouring, lattice rotation update

---

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `h5py`, `scikit-image`, `matplotlib`
- Optional: `cupy` + `cucim` (GPU), `pyvista` (VTK), `ipywidgets` + `ipympl` (Jupyter), `orix` (IPF)

# Installation

## Requirements

- Python ≥ 3.10

---

## Setting up an environment

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast, all-in-one Python package
and project manager.

**Install uv** (if you don't have it):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Create and activate the environment:**

```bash
# Create a virtual environment with Python 3.10
uv venv --python 3.10 .venv

# Activate — macOS / Linux
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

**Install the package:**

```bash
# Core only
uv pip install multimodal-registration

# With extras
uv pip install "multimodal-registration[vtk,jupyter,crystal]"

# GPU (pick your CUDA version)
uv pip install "multimodal-registration[cuda12,vtk,jupyter,crystal]"
```

!!! note "Working from source"
    If you cloned the repository, use `uv sync` instead — it reads
    `pyproject.toml` and installs everything declared there in one step:

    ```bash
    git clone https://github.com/damasres/multimodal-registration
    cd multimodal-registration
    uv sync --all-extras --group docs
    ```

---

### Using conda / mamba

[conda](https://docs.conda.io) (or the faster drop-in
[mamba](https://mamba.readthedocs.io)) works well when you also need
non-Python dependencies or want integration with an existing Anaconda setup.

**Create and activate the environment:**

```bash
conda create -n multimodal-reg python=3.10
conda activate multimodal-reg
```

**Install core dependencies from conda-forge** (recommended for compiled
packages such as h5py and scipy):

```bash
conda install -c conda-forge numpy scipy h5py scikit-image matplotlib
```

**Install the package with pip:**

```bash
pip install multimodal-registration

# With extras
pip install "multimodal-registration[vtk,jupyter,crystal]"
```

!!! note "CuPy on conda"
    For GPU support, install CuPy from conda-forge rather than PyPI —
    it links against the conda CUDA toolkit automatically:

    ```bash
    conda install -c conda-forge cupy cucim
    ```

**Save and recreate the environment:**

```bash
# Export
conda env export > environment.yml

# Recreate on another machine
conda env create -f environment.yml
```

---

## Optional extras

| Extra | Packages added | Required for |
|-------|---------------|--------------|
| `cuda11` / `cuda12` | `cupy`, `cucim` | GPU-accelerated zoom, shift, cross-correlation |
| `vtk` | `pyvista` | Loading `.vtk` / `.vts` / `.vtu` DVC files |
| `jupyter` | `ipywidgets`, `ipympl` | Interactive overlay in notebooks |
| `crystal` | `orix` | Rotating grain orientations and recomputing IPF colours |

---

## Verifying the install

```python
import multimodal_registration as mr
print(mr.backend_name())   # 'cuda' or 'cpu'
```

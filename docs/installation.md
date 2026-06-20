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

# GPU (CUDA 11)
uv pip install "multimodal-registration[cuda11,vtk,jupyter,crystal]"
```

!!! note "Working from source"
    If you cloned the repository, use `uv sync` instead — it reads
    `pyproject.toml` and installs everything declared there in one step:

    ```bash
    git clone https://github.com/pedrodamas0803/multimodal-registration
    cd multimodal-registration
    uv venv --prompt registration
    uv sync --all-extras
    ```

    The `uv venv --prompt registration` step creates the virtual environment
    with the shell prompt `(registration)`.  Skip it if you don't mind the
    default prompt (`multimodal-registration`).

**Register the Jupyter kernel** (requires the `jupyter` extra):

```bash
uv run python -m ipykernel install --user \
    --name registration \
    --display-name "Python (registration)"
```

After this, the kernel appears as **Python (registration)** in JupyterLab,
Jupyter Notebook, and VS Code. To remove it later:

```bash
jupyter kernelspec remove registration
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

### POWER9 / ppc64le (e.g. ESRF p9 cluster)

On `ppc64le` machines several PyPI packages have no pre-built wheels, so
**uv is not usable** on this architecture.  Use mamba with the provided
`environment-p9.yml` instead, which pins the versions known to work.

**Install micromamba** (no root needed):

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# or, if curl is unavailable:
wget -qO- micro.mamba.pm/install.sh | bash
```

**Create the environment** (user-local, no admin rights required):

```bash
mamba env create --prefix ~/.conda/envs/multimodal-reg \
    -f environment-p9.yml
mamba activate ~/.conda/envs/multimodal-reg
```

**Install the project in editable mode:**

```bash
pip install -e /path/to/multimodal-registration --no-deps
```

**GPU support** — CuPy must come from conda-forge (no ppc64le wheels on PyPI),
and cucim is unavailable on this architecture:

```bash
# Check your CUDA version first
nvidia-smi | head -3

mamba install -c conda-forge cupy -y
```

**VTK / PyVista** — also conda-forge only on ppc64le:

```bash
mamba install -c conda-forge pyvista -y
```

**Register the Jupyter kernel:**

```bash
python -m ipykernel install --user \
    --name multimodal-reg-p9 \
    --display-name "multimodal-reg (p9 GPU)"
```

??? note "Version constraints on ppc64le"
    - **ipympl** is pinned to `0.9.8`. Higher versions are not available on
      conda-forge for this architecture. If your JupyterHub shows a
      *"module not registered"* warning, it means the server's
      `jupyter-matplotlib` extension version does not match — ask your
      sysadmin to update it, or keep the pin and ignore the warning.
    - **matplotlib** is pinned to `<3.9`. Matplotlib 3.9+ changed internal
      event dispatching in a way that breaks interactive click events with
      ipympl 0.9.x.
    - **h5py** and **scipy** must come from conda-forge because the system
      HDF5 library on ppc64le nodes is typically too old to build h5py from
      source.

---

## Optional extras

| Extra | Packages added | Required for |
|-------|---------------|--------------|
| `cuda11` | `cupy`, `cucim` | GPU-accelerated zoom, shift, cross-correlation |
| `vtk` | `pyvista` | Loading `.vtk` / `.vts` / `.vtu` DVC files |
| `jupyter` | `ipywidgets`, `ipympl` | Interactive overlay in notebooks |
| `crystal` | `orix` | Rotating grain orientations and recomputing IPF colours |

---

## Verifying the install

```python
import multimodal_registration as mr
print(mr.backend_name())   # 'cuda' or 'cpu'
```

# GPU Acceleration

The package detects available hardware at import time and dispatches
compute-heavy operations to the fastest available backend.

---

## Backend priority

```
CUDA (CuPy)  →  CPU (NumPy / SciPy)
```

Apple MPS (Metal) does not yet expose a NumPy-compatible `ndimage` API
suitable for zoom and shift operations, so it falls through to CPU.

## Checking the active backend

```python
import multimodal_registration as mr
print(mr.backend_name())   # 'cuda' or 'cpu'
```

Or from the CLI:

```bash
multimodal-registration
# multimodal-registration  [backend: cuda]
```

---

## Installing CuPy

Pick the wheel that matches your installed CUDA toolkit:

```bash
# CUDA 12.x
pip install "multimodal-registration[cuda12]"

# CUDA 11.x
pip install "multimodal-registration[cuda11]"
```

Verify that CuPy can see your GPU:

```python
import cupy as cp
cp.zeros(1)          # raises if no GPU is found
print(cp.cuda.runtime.getDeviceCount())
```

---

## What runs on GPU

| Operation | CPU | GPU |
|-----------|-----|-----|
| `dct.upscale()` | `scipy.ndimage.zoom` | `cupyx.scipy.ndimage.zoom` |
| `dct.shift()` | `scipy.ndimage.shift` | `cupyx.scipy.ndimage.shift` |
| `pct.mask` binary closing | `scipy.ndimage.binary_closing` | `cupyx.scipy.ndimage.binary_closing` |
| `register()` cross-correlation | `skimage.registration.phase_cross_correlation` | `cucim.skimage.registration.phase_cross_correlation` |

The deformation gradient computation (`numpy.gradient`, `numpy.linalg.svd`),
`map_coordinates` warping, and IPF colour computation (orix) run on CPU
regardless of backend — these are either not bottlenecks or not yet supported
by CuPy/cuCIM.

---

## Memory considerations

Upscaling a DCT volume by a factor of ~3× in each dimension increases memory
by ~27×.  For large datasets:

- Process one volume at a time if GPU VRAM is limited.
- The `to_device` / `to_numpy` helpers in `backends.py` move arrays
  explicitly — you can stage data manually if needed.

```python
from multimodal_registration import backends

d_arr = backends.to_device(my_numpy_array)   # send to GPU
result = backends.to_numpy(d_arr)            # retrieve
```

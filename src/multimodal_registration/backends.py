"""
GPU/CPU backend detection and transparent dispatch.

Priority: CUDA (CuPy) > CPU (NumPy / SciPy)

Apple MPS (PyTorch) does not expose a NumPy-compatible ndimage API suited
for zoom/shift operations, so it falls through to CPU automatically.
Install cupy (e.g. `pip install cupy-cuda12x`) to enable CUDA acceleration.
"""
from __future__ import annotations

import warnings


def _try_cuda():
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpnd

        cp.zeros(1)  # trigger driver/context init — raises if no GPU
        return "cuda", cp, cpnd
    except Exception:
        return None


def _cpu_backend():
    import numpy as np
    import scipy.ndimage as spnd

    return "cpu", np, spnd


def _detect():
    result = _try_cuda()
    if result is not None:
        return result
    return _cpu_backend()


_BACKEND, _XP, _NDIMAGE = _detect()


def backend_name() -> str:
    """Return the active backend name: ``'cuda'`` or ``'cpu'``."""
    return _BACKEND


def xp():
    """Return the active array module (``cupy`` or ``numpy``)."""
    return _XP


def ndimage():
    """Return the active ndimage module (``cupyx.scipy.ndimage`` or ``scipy.ndimage``)."""
    return _NDIMAGE


def to_numpy(arr):
    """Convert an array to NumPy regardless of the active backend."""
    if _BACKEND == "cuda":
        return _XP.asnumpy(arr)
    import numpy as np

    return np.asarray(arr)


def to_device(arr):
    """Send a NumPy array to the active compute device (no-op on CPU)."""
    if _BACKEND == "cuda":
        return _XP.asarray(arr)
    return arr

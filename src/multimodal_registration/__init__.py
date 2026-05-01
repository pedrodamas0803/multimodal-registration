from .backends import backend_name
from .dct import DCT
from .deformation import apply_dvc, update_ipf, warp_dct
from .dvc import DVC, DVCMesh
from .interactive import ManualRegistration
from .io import write, write_dream3d, write_h5, write_vtk
from .pct import ReferencePCT
from .registration import find_shift, overlay_check, register

__all__ = [
    "DCT",
    "DVC",
    "DVCMesh",
    "ManualRegistration",
    "ReferencePCT",
    "apply_dvc",
    "warp_dct",
    "update_ipf",
    "find_shift",
    "register",
    "overlay_check",
    "write",
    "write_h5",
    "write_vtk",
    "write_dream3d",
    "backend_name",
]


def main() -> None:
    print(f"multimodal-registration  [backend: {backend_name()}]")

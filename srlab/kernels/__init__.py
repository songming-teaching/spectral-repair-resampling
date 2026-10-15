"""srlab.kernels -- GPU kernels.  One kernel lives here: the Stieltjes fixed-point iteration.

Nothing else is GPU-eligible (measured: dense fp64 matmul is 0.5x on this part, eigh 4.4x and never a
bottleneck).  See srlab/gpu.py for the measurements and the gate.
"""
from . import stieltjes       # noqa: F401

__all__ = ["stieltjes"]

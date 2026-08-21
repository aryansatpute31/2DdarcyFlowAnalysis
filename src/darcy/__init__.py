"""2D steady-state Darcy flow surrogates on the PDEBench benchmark.

PDE (PDEBench, `2D_DarcyFlow_beta*_Train.hdf5`):

    -div( a(x) grad u(x) ) = f      on  Omega = (0,1)^2
                      u(x) = 0      on  dOmega

with a constant forcing f = beta.  The task is the forward surrogate
a(x) -> u(x): given the permeability / diffusion coefficient field, predict the
pressure (hydraulic head) field.
"""

__all__ = ["config", "data", "metrics", "models", "physics", "utils"]

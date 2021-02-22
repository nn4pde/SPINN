# Fourier SPINN
# Solve differential equation u''(x) + 1 = 0 on (0,1)
# with Dirichlet boundar conditions u(0) = u(1) = 0.

import numpy as np

from common import tensor
from spinn1d import Plotter1D, App1D
from fourier1d_base import Fourier1D, FourierPlotter1D
from ode3 import ODESimple

if __name__ == '__main__':
    app = App1D(
        pde_cls=ODESimple, 
        nn_cls=Fourier1D,
        plotter_cls=FourierPlotter1D
    )
    app.run(modes=25, samples=100, lr=1e-3)
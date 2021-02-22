# Wavelet SPINN
# Solve differential equation u''(x) + 1 = 0 on (0,1)
# with Dirichlet boundar conditions u(0) = u(1) = 0.

import numpy as np

from common import tensor
from spinn1d import Plotter1D, App1D
from wavelet1d_base import Wavelet1D, WaveletPlotter1D
from ode1 import ODESimple

if __name__ == '__main__':
    app = App1D(
        pde_cls=ODESimple, 
        nn_cls=Wavelet1D,
        plotter_cls=WaveletPlotter1D
    )
    app.run(n_scale=5, n_shift=5, lr=1e-3)
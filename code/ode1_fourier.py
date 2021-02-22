# Fourier SPINN
# Solve differential equation u''(x) + 1 = 0 on (0,1)
# with Dirichlet boundar conditions u(0) = u(1) = 0.

import numpy as np

from common import tensor
from spinn1d import Plotter1D, App1D
from fourier1d_base import Fourier1D, FourierPlotter1D
from ode_base import BasicODE

class ODESimple(BasicODE):
    def pde(self, x, u, ux, uxx):
        return uxx + 1.0

    def has_exact(self):
        return True

    def exact(self, x):
        return 0.5*x*(1.0 - x)

    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = tensor(self.exact(self.xbn))
        bc = u - ub
        return (bc**2).sum()

    def plot_points(self):
        n = 25
        x = np.linspace(0.0, 1.0, n)
        return x

if __name__ == '__main__':
    app = App1D(
        pde_cls=ODESimple, 
        nn_cls=Fourier1D,
        plotter_cls=FourierPlotter1D
    )
    app.run(nodes=3, samples=15, lr=1e-2)

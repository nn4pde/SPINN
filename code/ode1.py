# Solve differential equation u''(x) + 1 = 0 on (0,1)
# with Dirichlet boundar conditions u(0) = u(1) = 0.

import numpy as np
import torch

from common import tensor
from spinn1d import Plotter1D, SPINN1D, App1D
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
        # return torch.abs(bc).max()

    def plot_points(self):
        n = 25
        x = np.linspace(0.0, 1.0, n)
        return x

if __name__ == '__main__':
    app = App1D(
        pde_cls=ODESimple, nn_cls=SPINN1D,
        plotter_cls=Plotter1D
    )
    app.run(nodes=3, samples=19, lr=1e-4, n_train=10000, tol=2.5e-5)

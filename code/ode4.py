# Solve differential equation u''(x) + f(x) = 0 on (0,1)
# where f = -4 pi^2 sin(2 pi x) + 250 sin(50 pi x)
# with Dirichlet boundar conditions u(0) = u(1) = 0.

import numpy as np
import torch

from common import tensor
from spinn1d import Plotter1D, SPINN1D, App1D
from ode_base import BasicODE


class ODE4(BasicODE):
    def pde(self, x, u, ux, uxx):
        pi = np.pi
        f = -pi*pi*(4*torch.sin(2*pi*x) + 250*torch.sin(50*pi*x))
        return uxx - f

    def has_exact(self):
        return True

    def exact(self, x):
        pi = np.pi
        f = np.sin(2*pi*x) + 0.1*np.sin(50*pi*x)
        return f

    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = tensor(self.exact(self.xbn))
        bc = u - ub
        return 1000*(bc**2).sum()

    def plot_points(self):
        n = 800
        x = np.linspace(0.0, 1.0, n)
        return x


if __name__ == '__main__':
    app = App1D(
        pde_cls=ODE4,
        nn_cls=SPINN1D,
        plotter_cls=Plotter1D
    )
    app.run(nodes=100, samples=800, n_train=20000,
            lr=2e-3, tol=1e-3)

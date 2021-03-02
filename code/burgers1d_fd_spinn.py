# Hybrid Finite Difference SPINN algorithm for the 1d Burgers equation

import numpy as np
import torch

from spinn1d import SPINN1D
from fd_spinn1d_base import FDSPINN1D, AppFD1D, FDPlotter1D


class Burgers1D(FDSPINN1D):
    def initial(self, x):
        u0 = torch.sin(2.0*np.pi*x)
        return u0

    def pde(self, x, u, ux, uxx, u0, dt):
        return -dt*u*ux - u + u0

    def has_exact(self):
        return False

    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = 0.0
        bc = u - ub
        return 100*(bc**2).sum()

    def plot_points(self):
        n = min(2*self.ns, 500)
        x = np.linspace(0.0, 1.0, n)
        return x


if __name__ == '__main__':
    app = AppFD1D(
        pde_cls=Burgers1D,
        nn_cls=SPINN1D,
        plotter_cls=FDPlotter1D
    )
    app.run(nodes=40, samples=400, dt=0.01, tol=5e-6, lr=1e-4,
            n_train=5000, n_skip=500)

# Hybrid Finite Difference SPINN algorithm for the 1d heat equation

import os
import numpy as np
import torch

from spinn1d import SPINN1D
from fd_spinn1d_base import FDSPINN1D, AppFD1D, FDPlotter1D


class Heat1D(FDSPINN1D):
    def initial(self, x):
        B1 = 2.0
        u0 = B1*torch.sin(np.pi*x)
        return u0

    def pde(self, x, u, ux, uxx, u0, dt):
        c = 1.0
        return c*c*self.dt*uxx - u + self.u0

    def has_exact(self):
        return True

    def exact(self, x):
        c = 1.0
        B1 = 2.0
        u = B1*np.exp(-np.pi*np.pi*c*c*self.t)*np.sin(np.pi*x)
        return u

    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = 0.0
        bc = u - ub
        return 100*(bc**2).sum()

    def plot_points(self):
        n = 50
        x = np.linspace(0.0, 1.0, n)
        return x


if __name__ == '__main__':
    app = AppFD1D(
        pde_cls=Heat1D,
        nn_cls=SPINN1D,
        plotter_cls=FDPlotter1D
    )
    app.run()

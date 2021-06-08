# Hybrid Finite Difference SPINN algorithm for the 1d Burgers equation

import numpy as np
import torch

from common import tensor
from spinn1d import SPINN1D
from fd_spinn1d_base import FDSPINN1D, AppFD1D, FDPlotter1D


class Advection1D(FDSPINN1D):
    def __init__(self, n, ns, sample_frac=1.0, dt=1e-3, T=1.0, t_skip=10):
        super().__init__(n, ns, sample_frac, dt=dt, T=T, t_skip=t_skip)

        self.xn = np.asarray([-1.0 + 2*i/(n + 1) for i in range(1, (n + 1))])
        self.xs = tensor(
            [-1.0 + 2*i/(ns + 1) for i in range(1, (ns + 1))],
            requires_grad=True
        )

        self.xbn = np.array([-1.0, 1.0])
        self.xb = tensor(self.xbn)
        self.u0 = self.initial(self.xs)

    def initial(self, x):
        z = (x + 0.3)/0.15
        u0 = torch.exp(-0.5*z**2)
        return u0

    def pde(self, x, u, ux, uxx, u0, dt):
        a = 0.5
        return -dt*a*ux - u + u0

    def has_exact(self):
        return True

    def exact(self, x):
        a = 0.5
        z = (x - a*self.t + 0.3)/0.15
        return np.exp(-0.5*z**2)

    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = 0.0
        bc = u - ub
        return 100*(bc**2).sum()

    def plot_points(self):
        n = min(2*self.ns, 500)
        x = np.linspace(-1.0, 1.0, n)
        return x


if __name__ == '__main__':
    app = AppFD1D(
        pde_cls=Advection1D,
        nn_cls=SPINN1D,
        plotter_cls=FDPlotter1D
    )
    app.run(nodes=20, samples=500, dt=0.0025,
            tol=1e-6, lr=1e-4, n_train=5000)

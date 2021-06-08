# Hybrid Finite Difference SPINN algorithm for the 1d Burgers equation

import numpy as np
import torch

from common import tensor
from spinn1d import SPINN1D
from fd_spinn1d_base import FDSPINN1D, AppFD1D, FDPlotter1D


class AllenCahnFD(FDSPINN1D):
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
        u0 = x*x*torch.cos(np.pi*x)
        return u0

    def pde(self, x, u, ux, uxx, u0, dt):
        a = 0.0001
        return u - u0 + dt*(-a*uxx + 5*u**3 - 5*u)

    def has_exact(self):
        return False

    def exact(self, x, t=0.0):
        return x*x*np.cos(np.pi*x)

    def boundary_loss(self, nn):
        xb = tensor(self.boundary(), requires_grad=True)
        u = nn(xb)
        _, ux, _ = self._compute_derivatives(u, xb)
        bc_u = u[0] - u[1]
        bc_ux = ux[0] - ux[1]
        return (bc_u**2 + bc_ux**2).sum()

    def old_boundary_loss(self, nn):
        xb = self.boundary()
        u = nn(xb)
        ub = -1.0
        bc = u - tensor(ub)
        return (bc**2).sum()

    def plot_points(self):
        n = min(2*self.ns, 500)
        x = np.linspace(-1.0, 1.0, n)
        return x


if __name__ == '__main__':
    app = AppFD1D(
        pde_cls=AllenCahnFD,
        nn_cls=SPINN1D,
        plotter_cls=FDPlotter1D
    )
    app.run(nodes=100, samples=500, dt=0.005,
            tol=1e-6, lr=1e-3, n_train=5000)

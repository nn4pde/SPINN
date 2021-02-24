# Variational implementation of SPINN for simple ODE
# uxx + 1 = 0

import numpy as np
import torch

from common import tensor
from spinn1d import Plotter1D, SPINN1D, App1D
from var1d_base import Var1D

class ODESimple(Var1D):
    def _f(self, x):
        K = 0.01
        f = -(K*(-6*x + 4/3.) +
              x*(2*x - 2./3)**2)*torch.exp(-(x - 1/3.)**2/K)/K**2
        return f

    def pde(self, x, u, ux, uxx):
        return 0.5*ux*ux - self._f(x)*u

    def has_exact(self):
        return True

    def exact(self, x):
        K = 0.01
        return x*(np.exp(-((x - (1.0/3.0))**2)/K) -
                  np.exp(-4.0/(9.0*K)))

    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = tensor(self.exact(self.xbn))
        bc = u - ub
        return (bc**2).sum()

    def plot_points(self):
        n = 50
        x = np.linspace(0.0, 1.0, n)
        return x

if __name__ == '__main__':
    app = App1D(
        pde_cls=ODESimple, nn_cls=SPINN1D,
        plotter_cls=Plotter1D
    )
    app.run(nodes=3, samples=15, lr=1e-2, tol=-10.0)

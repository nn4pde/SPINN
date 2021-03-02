# Variational implementation of SPINN for simple ODE
# uxx + 1 = 0

import numpy as np

from common import tensor
from spinn1d import Plotter1D, SPINN1D, App1D
from var1d_base import Var1D

class ODESimple(Var1D):
    def pde(self, x, u, ux, uxx):
        return 0.5*ux*ux - u

    def has_exact(self):
        return True

    def exact(self, x):
        return 0.5*x*(1.0 - x)

    # def interior_loss(self, nn):
    #     res = self._get_residue(nn)
    #     return res.sum()

    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = tensor(self.exact(self.xbn))
        bc = u - ub
        return 1000*(bc**2).sum()

    def plot_points(self):
        n = 25
        x = np.linspace(0.0, 1.0, n)
        return x

if __name__ == '__main__':
    app = App1D(
        pde_cls=ODESimple, nn_cls=SPINN1D,
        plotter_cls=Plotter1D
    )
    app.run(nodes=5, samples=500, n_train=5000, lr=1e-3, tol=-10.0)

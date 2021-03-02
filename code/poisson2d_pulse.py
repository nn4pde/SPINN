# Solve Poisson equation in 2D using SPINN
# \nabla^2 u(x, y) = f(x,y) on [0,1]x[0,1]
# (See pde() method below for exact form of f)
# Zero Dirichlet boundary condition on boundary

import numpy as np
import torch

from common import tensor
from spinn2d import Plotter2D, SPINN2D, App2D
from pde2d_base import RegularPDE

PI = np.pi

class Poisson2D(RegularPDE):
    def pde(self, x, y, u, ux, uy, uxx, uyy):
        K = 0.02
        ex = torch.exp(-(x - 0.25)*(x - 0.25)/K)
        fxx = (
            (1.0 + ((1.0 - 2.0*x)*(x - 0.25) + x*(1.0 - x))/K)
            + ((1.0 - 2.0*x - 2.0*x*(1 - x)*(x - 0.25)/K)*(x - 0.25)/K)
        )*2.0*ex*y*(1 - y)
        fyy = 2.0*x*(1.0 - x)*ex
        return uxx + uyy + fxx + fyy

    def has_exact(self):
        return True

    def exact(self, x, y):
        K = 0.02
        return x*(1.0 - x)*y*(1.0 - y)*np.exp(-(x - 0.25)*(x - 0.25)/K)

    def boundary_loss(self, nn):
        xb, yb = self.boundary()
        xbn, ybn = (t.detach().cpu().numpy() for t in (xb, yb))

        u = nn(xb, yb)
        ub = tensor(self.exact(xbn, ybn))
        bc = u - ub
        return (bc**2).sum()

if __name__ == '__main__':
    app = App2D(
        pde_cls=Poisson2D, nn_cls=SPINN2D,
        plotter_cls=Plotter2D
    )
    app.run(nodes=100, samples=400, n_train=25000, lr=1e-3, tol=1e-3)
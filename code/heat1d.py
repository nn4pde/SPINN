# Heat equation: 1D
# Treated as 2d spacetime PDE

import numpy as np

from common import tensor
from ibvp2d_base import IBVP2D
from spinn2d import Plotter2D, SPINN2D, App2D

class Heat1D(IBVP2D):
    def pde(self, x, t, u, ux, ut, uxx, utt):
        c = 1.0
        return ut - c*c*uxx
    
    def has_exact(self):
        return True

    def exact(self, x, t):
        b1 = 2.0
        c  = 1.0
        L  = self.xR - self.xL
        a2 = (np.pi*c/L)**2
        return b1*np.exp(-a2*t)*np.sin(np.pi*x/L)

    def boundary_loss(self, nn):
        xb, tb =self.boundary()
        u = nn(xb, tb)
        ub = self.exact(xb.detach().cpu().numpy(), tb.detach().cpu().numpy())
        bc = u - tensor(ub)
        return (bc**2).sum()

if __name__ == '__main__':
    app = App2D(
        pde_cls=Heat1D,
        nn_cls=SPINN2D,
        plotter_cls=Plotter2D
    )
    app.run(nodes=50, samples=200, lr=1e-2)


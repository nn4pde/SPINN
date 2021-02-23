# Heat equation: 1D
# Treated as 2d spacetime PDE

import os
import numpy as np
import torch

from common import tensor
from ibvp2d_base import IBVP2D
from spinn2d import Plotter2D, SPINN2D, App2D


class HeatPlotter(Plotter2D):
    def save(self, dirname):
        '''Save the model and results.

        '''
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(self.nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz')
        xp, yp, up = self.get_plot_data()
        u_exact = self.pde.exact(xp, yp)
        x, t = np.mgrid[0:1:100j, 0:0.2:21j]
        xt, tt = tensor(x.ravel()), tensor(t.ravel())
        u = self.nn(xt, tt).detach().cpu().numpy()
        u.shape = x.shape
        np.savez(
            rfile, xp=xp, yp=yp, up=up, u_exact=u_exact,
            x=x, t=t, u=u
        )


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
        xb, tb = self.boundary()
        u = nn(xb, tb)
        ub = self.exact(xb.detach().cpu().numpy(), tb.detach().cpu().numpy())
        bc = u - tensor(ub)
        return (bc**2).sum()


if __name__ == '__main__':
    app = App2D(
        pde_cls=Heat1D,
        nn_cls=SPINN2D,
        plotter_cls=HeatPlotter
    )
    app.run(nodes=50, samples=200, lr=1e-2)

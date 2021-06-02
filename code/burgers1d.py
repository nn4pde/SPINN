# Burgers equation: 1D
# Treated as 2d spacetime PDE

import os
import numpy as np
import torch

from common import tensor
from ibvp2d_base import IBVP2D
from spinn2d import Plotter2D, SPINN2D, App2D


class Burgers1D(IBVP2D):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples,
                   args.xL, args.xR, args.T,
                   args.ic, args.viscosity,
                   args.sample_frac)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        super().setup_argparse(parser, **kw)
        p = parser
        p.add_argument(
            '--viscosity', dest='viscosity',
            default=kw.get('viscosity', 0.0), type=float,
            help='Viscosity in viscous Burgers equation.'
        )
        p.add_argument(
            '--ic', dest='ic', default=kw.get('ic', 'gaussian'),
            choices=['jump', 'gaussian', 'sin', 'sin2'],
            help='Initial profiles for linear advection equation.'
        )

    def __init__(self, n, ns, nb=None, nbs=None, xL=0.0, xR=1.0, T=1.0,
                 ic='gaussian', viscosity=0.0, sample_frac=1.0):
        super().__init__(n, ns, nb, nbs, xL, xR, T, sample_frac)
        self.ic = ic
        self.viscosity = viscosity

    def pde(self, x, t, u, ux, ut, uxx, utt):
        return ut + u*ux - self.viscosity*uxx

    def interior_loss(self, nn):
        res = self._get_residue(nn)
        r = (res**2).mean()
        return r

    def has_exact(self):
        return False

    def _get_residue_plot(self, nn, xs, ys):
        u = nn(xs, ys)
        u, ux, uy, uxx, uyy = self._compute_derivatives(u, xs, ys)
        res = self.pde(xs, ys, u, ux, uy, uxx, uyy)
        return res

    def _boundary_condition(self, x, t):
        a = 0.0
        if self.ic == 'jump':
            z = (x - a*t + 1.1)
            return np.heaviside(z, 0.5) - 2*np.heaviside(z - 1.1, 0.5)
        elif self.ic == 'gaussian':
            z = (x - a*t + 0.3)/0.15
            return np.exp(-0.5*z**2)
        elif self.ic == 'sin':
            z = (x - a*t + 0.75)
            y = np.heaviside(z, 0.5) - np.heaviside(z - 1.0, 0.5)
            return y*np.sin(z*np.pi)
        elif self.ic == 'sin2':
            z = (x - a*t + 0.5)
            y = np.heaviside(z, 0.5) - np.heaviside(z - 1.0, 0.5)
            return y*np.sin(z*np.pi*2)

    def boundary_loss(self, nn):
        xb, tb = self.boundary()
        u = nn(xb, tb)
        ub = self._boundary_condition(
            xb.detach().cpu().numpy(), tb.detach().cpu().numpy()
        )
        bc = u - tensor(ub)
        l = (bc**2).sum()
        return l


class MyPlotter(Plotter2D):
    def save(self, dirname):
        '''Save the model and results.

        '''
        nn = self.nn
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz')
        x, t = np.mgrid[-1:1:500j, 0:1:11j]
        xt, tt = tensor(x.ravel()), tensor(t.ravel())
        u = nn(xt, tt).detach().cpu().numpy()
        u.shape = x.shape
        xp, tp, up = self.get_plot_data()
        np.savez(rfile, x=x, t=t, u=u, xp=xp, tp=tp, up=up)


if __name__ == '__main__':
    app = App2D(
        pde_cls=Burgers1D,
        nn_cls=SPINN2D,
        plotter_cls=MyPlotter
    )
    app.run(nodes=200, samples=1000, xL=-1.0, xR=1.0, lr=1e-3)

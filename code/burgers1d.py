# Burgers equation: 1D
# Treated as 2d spacetime PDE

import numpy as np

from common import tensor
from ibvp2d_base import IBVP2D
from spinn2d import Plotter2D, SPINN2D, App2D

class Burgers1D(IBVP2D):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples,
                   args.xL, args.xR, args.T,
                   args.ic, args.viscosity)

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
            choices=['hat', 'gaussian', 'sin', 'sin2'],
            help='Initial profiles for linear advection equation.'
        )

    def __init__(self, n, ns, 
        nb=None, nbs=None, 
        xL=0.0, xR=1.0, T=1.0, 
        ic='gaussian', viscosity=0.0):

        super().__init__(n, ns, nb, nbs, xL, xR, T)
        self.ic = ic
        self.viscosity = viscosity

    def pde(self, x, t, u, ux, ut, uxx, utt): 
        return ut + u*ux - self.viscosity*uxx
    
    def has_exact(self):
        return False

    def _boundary_condition(self, x, t):
        a = 0.5
        if self.ic == 'hat':
            z = (x - a*t + 0.35)
            return np.heaviside(z, 0.5) - np.heaviside(z - 0.5, 0.5)
        elif self.ic == 'gaussian':
            z = (x - a*t + 0.3)/0.15
            return np.exp(-z**2)
        elif self.ic == 'sin':
            z = (x - a*t + 0.5)
            y = np.heaviside(z, 0.5) - np.heaviside(z - 0.5, 0.5)
            return np.sin(z*np.pi*2*y)
        elif self.ic == 'sin2':
            z = (x - a*t + 0.5)
            y = np.heaviside(z, 0.5) - np.heaviside(z - 0.5, 0.5)
            return np.sin(z*np.pi*4*y)

    def boundary_loss(self, nn):
        xb, tb =self.boundary()
        u = nn(xb, tb)
        ub = self._boundary_condition(xb.detach().cpu().numpy(), tb.detach().cpu().numpy())
        bc = u - tensor(ub)
        return (bc**2).sum()

if __name__ == '__main__':
    app = App2D(
        pde_cls=Burgers1D,
        nn_cls=SPINN2D,
        plotter_cls=Plotter2D
    )
    app.run(nodes=100, samples=400, xL=-1.0, xR=1.0, lr=1e-2)


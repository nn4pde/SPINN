# Basic PDEs : 2D

import numpy as np
import torch
import torch.autograd as ag
from common import PDE, tensor
from spinn2d import Plotter2D, SPINN2D, App2D


class RegularPDE(PDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples,
                   args.sample_frac)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--nodes', '-n', dest='nodes',
            default=kw.get('nodes', 25), type=int,
            help='Number of nodes to use.'
        )
        p.add_argument(
            '--samples', '-s', dest='samples',
            default=kw.get('samples', 100), type=int,
            help='Number of sample points to use.'
        )
        p.add_argument(
            '--b-nodes', dest='b_nodes',
            default=kw.get('b_nodes', None), type=int,
            help='Number of boundary nodes to use per edge'
        )
        p.add_argument(
            '--b-samples', dest='b_samples',
            default=kw.get('b_samples', None), type=int,
            help='Number of boundary samples to use per edge.'
        )
        p.add_argument(
            '--sample-frac', '-f', dest='sample_frac',
            default=kw.get('sample_frac', 1.0), type=float,
            help='Fraction of interior nodes used for sampling.'
        )

    def _compute_derivatives(self, u, xs, ys):
        du = ag.grad(
            outputs=u, inputs=(xs, ys), grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )
        d2ux = ag.grad(
            outputs=du[0], inputs=(xs, ys),
            grad_outputs=torch.ones_like(du[0]),
            retain_graph=True, create_graph=True
        )
        d2uy = ag.grad(
            outputs=du[1], inputs=(xs, ys),
            grad_outputs=torch.ones_like(du[1]),
            retain_graph=True, create_graph=True
        )

        return u, du[0], du[1], d2ux[0],  d2uy[1]

    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):
        self.sample_frac = sample_frac

        # Interior nodes
        n = round(np.sqrt(n_nodes) + 0.49)
        self.n = n
        dxb2 = 0.5/(n + 1)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, n*1j)
        x, y = np.mgrid[sl, sl]
        self.i_nodes = (x.ravel(), y.ravel())

        # Fixed nodes
        nb = n if nb is None else nb
        self.nb = nb
        dxb2 = 0.5/(nb)
        _x = np.linspace(dxb2, 1.0 - dxb2, nb)
        _o = np.ones_like(_x)
        x = np.hstack((_x, _o, _x, 0*_o))
        y = np.hstack((_o*0, _x, _o, _x))
        self.f_nodes = (x, y)

        # Interior samples
        self.ns = ns = round(np.sqrt(ns) + 0.49)
        dxb2 = 0.5/(ns)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, ns*1j)
        xs, ys = (tensor(t.ravel(), requires_grad=True)
                  for t in np.mgrid[sl, sl])
        self.p_samples = (xs, ys)

        self.n_interior = len(self.p_samples[0])
        self.rng_interior = np.arange(self.n_interior)
        self.sample_size = int(self.sample_frac*self.n_interior)

        # Boundary samples
        nbs = ns if nbs is None else nbs
        self.nbs = nbs
        sl = slice(0.0, 1.0, nbs*1j)
        x, y = np.mgrid[sl, sl]
        cond = ((x < xl) | (x > xr)) | ((y < xl) | (y > xr))
        xb, yb = (tensor(t.ravel(), requires_grad=True)
                  for t in (x[cond], y[cond]))
        self.b_samples = (xb, yb)

    def nodes(self):
        return self.i_nodes

    def fixed_nodes(self):
        return self.f_nodes

    def interior(self):
        if abs(self.sample_frac - 1.0) < 1e-3:
            return self.p_samples
        else:
            idx = np.random.choice(self.rng_interior, 
                size=self.sample_size, replace=False)
            x, y = self.p_samples
            return x[idx], y[idx]

    def boundary(self):
        return self.b_samples

    def plot_points(self):
        n = self.ns*2
        x, y = np.mgrid[0:1:n*1j, 0:1:n*1j]
        return x, y

    def _get_residue(self, nn):
        xs, ys = self.interior()
        u = nn(xs, ys)
        u, ux, uy, uxx, uyy = self._compute_derivatives(u, xs, ys)
        res = self.pde(xs, ys, u, ux, uy, uxx, uyy)
        return res

    def interior_loss(self, nn):
        res = self._get_residue(nn)
        return (res**2).mean()
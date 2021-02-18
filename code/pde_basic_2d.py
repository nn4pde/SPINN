# Basic PDEs : 2D

import numpy as np
import torch
from common import PDE, tensor
from spinn2d import Problem2D, SPINN2D, App2D

PI = np.pi


class RegularPDE(PDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples)

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

    def __init__(self, n_nodes, ns, nb=None, nbs=None):
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
        return self.p_samples

    def boundary(self):
        return self.b_samples

    def plot_points(self):
        n = self.ns*2
        x, y = np.mgrid[0:1:n*1j, 0:1:n*1j]
        return x, y

    def eval_bc(self, problem):
        xb, yb = self.boundary()
        xbn, ybn = (t.detach().cpu().numpy() for t in (xb, yb))

        u = problem.nn(xb, yb)
        ub = tensor(self.exact(xbn, ybn))
        return u - ub

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        raise NotImplementedError()

    def exact(self, x, y):
        pass


class ExactPDE(RegularPDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples, args.de)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        super().setup_argparse(parser, **kw)
        p = parser
        p.add_argument(
            '--de', dest='de', default=kw.get('de', 'sin2'),
            choices=['sin2', 'bump'],
            help='Differential equation to solve.'
        )

    def __init__(self, n_nodes, ns, nb=None, nbs=None, de='bump'):
        super().__init__(n_nodes, ns, nb=nb, nbs=nbs)
        self.deq = de

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        if self.deq == 'sin2':
            f = 20*PI**2*torch.sin(2*PI*x)*torch.sin(4*PI*y)
            return 0.1*(uxx + uyy + f)
        elif self.deq == 'bump':
            K = 0.05
            ex = torch.exp(-(x - 0.25)*(x - 0.25)/K)
            fxx = (
                (1.0 + ((1.0 - 2.0*x)*(x - 0.25) + x*(1.0 - x))/K)
                + ((1.0 - 2.0*x - 2.0*x*(1 - x)*(x - 0.25)/K)*(x - 0.25)/K)
            )*2.0*ex*y*(1 - y)
            fyy = 2.0*x*(1.0 - x)*ex
            return uxx + uyy + fxx + fyy

    def exact(self, x, y):
        if self.deq == 'sin2':
            return np.sin(2*PI*x)*np.sin(4*PI*y)
        elif self.deq == 'bump':
            K = 0.05
            return x*(1.0 - x)*y*(1.0 - y)*np.exp(-(x - 0.25)*(x - 0.25)/K)


if __name__ == '__main__':
    app = App2D(
        problem_cls=Problem2D, nn_cls=SPINN2D,
        pde_cls=ExactPDE
    )
    app.run(nodes=40, samples=120, lr=1e-2)
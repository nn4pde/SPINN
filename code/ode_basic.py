# Basic ODEs (1D)

import numpy as np
import torch

from common import PDE, tensor
from spinn1d import Problem1D, SPINN1D, App1D

class ODE(PDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--nodes', '-n', dest='nodes',
            default=kw.get('nodes', 10), type=int,
            help='Number of nodes to use.'
        )
        p.add_argument(
            '--samples', '-s', dest='samples',
            default=kw.get('samples', 30), type=int,
            help='Number of sample points to use.'
        )

    def __init__(self, n, ns):
        self.n = n
        self.ns = ns

        self.xn = np.asarray([i/(n + 1) for i in range(1, (n + 1))])
        self.xs = tensor(
            [i/(ns + 1) for i in range(1, (ns + 1))],
            requires_grad=True
        )

        self.xbn = np.array([0.0, 1.0])
        self.xb = tensor(self.xbn)

    def nodes(self):
        return self.xn

    def fixed_nodes(self):
        return self.xbn

    def interior(self):
        return self.xs

    def boundary(self):
        return self.xb

    def plot_points(self):
        n = 26 #51
        x = np.linspace(0.0, 1.0, n, endpoint=True)
        return x

    def eval_bc(self, problem):
        u = problem.nn(self.boundary())
        ub = tensor(self.exact(self.xbn))
        return u - ub

    # ####### Override these for different differential equations
    def pde(self, x, u, ux, uxx):
        raise NotImplementedError()

    def exact(self, x, y):
        pass


class ExactODE(ODE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples, args.de)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        super().setup_argparse(parser, **kw)
        p = parser
        p.add_argument(
            '--de', dest='de', default=kw.get('de', 'sin8'),
            choices=['sin8', 'pulse', 'simple'],
            help='Differential equation to solve.'
        )

    def __init__(self, n, ns, deq):
        super().__init__(n, ns)
        self.deq = deq

    def pde(self, x, u, ux, uxx):
        if self.deq == 'pulse':
            K = 0.01
            f = -(K*(-6*x + 4/3.) +
                  x*(2*x - 2./3)**2)*torch.exp(-(x - 1/3.)**2/K)/K**2
            return uxx + f
        elif self.deq == 'sin8':
            return uxx + torch.sin(8.0*np.pi*x)
        elif self.deq == 'simple':
            return uxx + 1.0

    def exact(self, x):
        if self.deq == 'pulse':
            K = 0.01
            return x*(np.exp(-((x - (1.0/3.0))**2)/K) -
                      np.exp(-4.0/(9.0*K)))
        elif self.deq == 'sin8':
            return np.sin(8.0*np.pi*x)/(64.0*np.pi**2)
        elif self.deq == 'simple':
            return 0.5*x*(1.0 - x)


if __name__ == '__main__':
    app = App1D(
        problem_cls=Problem1D, nn_cls=SPINN1D,
        pde_cls=ExactODE
    )
    app.run(nodes=20, samples=80, lr=1e-2)
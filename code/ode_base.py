# Basic ODEs (1D)

import numpy as np
import torch
import torch.autograd as ag
from common import PDE, tensor

class BasicODE(PDE):
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

    def _compute_derivatives(self, u, x):
        du = ag.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True, allow_unused=True
        )

        d2u = ag.grad(
            outputs=du, inputs=x, grad_outputs=torch.ones_like(du[0]),
            retain_graph=True, create_graph=True, allow_unused=True
        )
        return u, du[0], d2u[0]

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
        n = 2*self.ns
        x = np.linspace(0.0, 1.0, n)
        return x

    def interior_loss(self, nn):
        xs = self.interior()
        u = nn(xs)
        u, ux, uxx = self._compute_derivatives(u, xs)
        res = self.pde(xs, u, ux, uxx)
        return (res**2).sum()
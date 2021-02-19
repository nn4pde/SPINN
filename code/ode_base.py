# Basic ODEs (1D)

import numpy as np
import torch
import torch.autograd as ag
from common import PDE, tensor

class BasicODE(PDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples, args.sample_frac)

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
        p.add_argument(
            '--sample-frac', dest='sample_frac',
            default=kw.get('sample_frac', 1.0), type=float,
            help='Fraction of interior nodes used for sampling.'
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

    def __init__(self, n, ns, sample_frac=1.0):
        self.n = n
        self.ns = ns
        self.sample_frac = sample_frac

        self.xn = np.asarray([i/(n + 1) for i in range(1, (n + 1))])
        self.xs = tensor(
            [i/(ns + 1) for i in range(1, (ns + 1))],
            requires_grad=True
        )

        self.xbn = np.array([0.0, 1.0])
        self.xb = tensor(self.xbn)

        self.rng_interior = np.arange(self.ns)
        self.sample_size = int(self.sample_frac*self.ns)

    def nodes(self):
        return self.xn

    def fixed_nodes(self):
        return self.xbn

    def interior(self):
        if abs(self.sample_frac - 1.0) < 1e-3:
            return self.xs
        else:
            idx = np.random.choice(self.rng_interior, 
                size=self.sample_size, replace=False)
            return self.xs[idx]

    def boundary(self):
        return self.xb

    def plot_points(self):
        n = 2*self.ns
        x = np.linspace(0.0, 1.0, n)
        return x

    def _get_residue(self, nn):
        xs = self.interior()
        u = nn(xs)
        u, ux, uxx = self._compute_derivatives(u, xs)
        res = self.pde(xs, u, ux, uxx)
        return res

    def interior_loss(self, nn):
        res = self._get_residue(nn)
        return (res**2).mean()
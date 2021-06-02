# Base class for initial and boundary value problems in 2D
# (1 space dimension and 1 time dimension)
# Common class to implement both parabolic and hyperbolic PDEs

import numpy as np
from numpy.testing._private.utils import requires_memory
import torch
import torch.autograd as ag
from common import PDE, tensor


class IBVP2D(PDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples,
                   args.b_nodes, args.b_samples,
                   args.xL, args.xR, args.T,
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
            help='Number of boundary nodes to use per edge.'
        )
        p.add_argument(
            '--b-samples', dest='b_samples',
            default=kw.get('b_samples', None), type=int,
            help='Number of boundary samples to use per edge.'
        )
        p.add_argument(
            '--xL', dest='xL',
            default=kw.get('xL', 0.0), type=float,
            help='Left end of domain.'
        )
        p.add_argument(
            '--xR', dest='xR',
            default=kw.get('xR', 1.0), type=float,
            help='Right end of domain.'
        )
        p.add_argument(
            '--duration', dest='T',
            default=kw.get('T', 1.0), type=float,
            help='Duration of simulation.'
        )
        p.add_argument(
            '--sample-frac', '-f', dest='sample_frac',
            default=kw.get('sample_frac', 1.0), type=float,
            help='Fraction of interior nodes used for sampling.'
        )

    def _compute_derivatives(self, u, xs, ts):
        du = ag.grad(
            outputs=u, inputs=(xs, ts),
            grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )
        d2ux = ag.grad(
            outputs=du[0], inputs=(xs, ts),
            grad_outputs=torch.ones_like(du[0]),
            retain_graph=True, create_graph=True
        )
        d2ut = ag.grad(
            outputs=du[1], inputs=(xs, ts),
            grad_outputs=torch.ones_like(du[1]),
            retain_graph=True, create_graph=True
        )

        return u, du[0], du[1], d2ux[0],  d2ut[1]

    def _get_points_split(self, L, T, n):
        size = np.sqrt(L*T/n)
        nx = round((L/size) + 0.49)
        nt = round((T/size) + 0.49)
        return nx, nt

    def _get_points_mgrid(self, xL, xR, T, nx, nt, endpoint=False):
        if endpoint:
            slx = slice(xL, xR, nx*1j)
            slt = slice(0.0, T, nt*1j)
            x,t = np.mgrid[slx, slt]
            return (x.ravel(), t.ravel())
        else:
            dxb2 = 0.5*(xR - xL)/(nx + 1)
            xl, xr = xL + dxb2, xR - dxb2
            slx = slice(xl, xr, nx*1j)
            dt = T/(nt + 1)
            slt = slice(dt, T, nt*1j)
            x, t = np.mgrid[slx, slt]
            return (x.ravel(), t.ravel())

    def _get_boundary_nodes(self, xL, xR, T, nx, nt):
        x1 = np.asarray([xL + i*(xR - xL)/(nx - 1) for i in range(nx)])
        t1 = np.zeros(nx)

        x2 = xL*np.ones(nt)
        t2 = np.asarray([j*T/nt for j in range(1, (nt + 1))])

        x3 = xR*np.ones(nt)
        t3 = np.asarray([j*T/nt for j in range(1, (nt + 1))])

        xs = np.concatenate((x1, x2, x3))
        ts = np.concatenate((t1, t2, t3))

        return (xs, ts)

    def __init__(self, n_nodes, ns, nb=None, nbs=None, xL=0.0, xR=1.0,
                 T=1.0, sample_frac=1.0):
        self.xL = xL
        self.xR = xR
        self.T  = T

        self.sample_frac = sample_frac

        # Interior nodes
        self.nx, self.nt = self._get_points_split((xR - xL), T, n_nodes)
        self.i_nodes = self._get_points_mgrid(xL, xR, T,
            self.nx, self.nt, endpoint=False)

        # Fixed nodes
        if nb is None:
            nbx, nbt = self.nx, self.nt
        else:
            nbx, nbt = self._get_points_split((xR - xL), T, nb + 1)

        if nb == 0:
            self.f_nodes = ([], [])
        else:
            self.f_nodes = self._get_boundary_nodes(xL, xR, T, nbx, nbt)

        # Interior samples
        self.nsx, self.nst = self._get_points_split((xR - xL), T, ns)
        xi, ti = self._get_points_mgrid(xL, xR, T,
            self.nsx, self.nst, endpoint=False)
        self.p_samples = (tensor(xi, requires_grad=True),
                          tensor(ti, requires_grad=True))

        self.n_interior = len(self.p_samples[0])
        self.rng_interior = np.arange(self.n_interior)
        self.sample_size = int(self.sample_frac*self.n_interior)

        # Boundary samples
        if nbs is None:
            nbsx, nbst = self.nsx, self.nst
        else:
            nbsx, nbst = self._get_points_split((xR - xL), T, nbs)

        xb, tb = self._get_boundary_nodes(xL, xR, T, nbsx, nbst)
        self.b_samples = (tensor(xb, requires_grad=True),
                          tensor(tb, requires_grad=True))

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
        x, t = np.mgrid[self.xL:self.xR:self.nsx*1j,
                        0.0:self.T:self.nst*1j]
        return x, t

    def _get_residue(self, nn):
        xs, ts = self.interior()
        u = nn(xs, ts)
        u, ux, ut, uxx, utt = self._compute_derivatives(u, xs, ts)
        res = self.pde(xs, ts, u, ux, ut, uxx, utt)
        return res

    def interior_loss(self, nn):
        res = self._get_residue(nn)
        return (res**2).mean()

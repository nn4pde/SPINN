# Poisson equation on a general domain

import os
import numpy as np
import torch
import torch.autograd as ag
from mayavi import mlab
from common import PDE, tensor
from spinn2d import Plotter2D, App2D, SPINN2D


class Poisson2D(PDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.f_nodes_int, args.f_nodes_bdy,
            args.f_samples_int, args.f_samples_bdy,
            args.sample_frac)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        d = os.path.abspath(os.path.dirname(__file__))
        def _f(s):
            return os.path.join(d, 'mesh_data', s)

        p.add_argument(
            '--f_nodes_int', '-ni', dest='f_nodes_int',
            default=kw.get('f_nodes_int', _f('interior_nodes.dat')),
            type=str,
            help='File containing interior nodes.'
        )
        p.add_argument(
            '--f_nodes_bdy', '-nb', dest='f_nodes_bdy',
            default=kw.get('f_nodes_bdy', _f('boundary_nodes.dat')),
            type=str,
            help='File containing boundary nodes.'
        )
        p.add_argument(
            '--f_samples_int', '-si', dest='f_samples_int',
            default=kw.get('f_samples_int', _f('interior_samples.dat')),
            type=str,
            help='File containing interior samples.'
        )
        p.add_argument(
            '--f_samples_bdy', '-sb', dest='f_samples_bdy',
            default=kw.get('f_samples_bdy', _f('boundary_samples.dat')),
            type=str,
            help='File containing boundary samples.'
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

    def _extract_coordinates(self, f_pts):
        xs = []
        ys = []

        with open(f_pts, 'r') as f:
            line = f.readline()
            words = line.split()
            n = int(words[0])

            for _ in range(n):
                line = f.readline()
                words = line.split()
                xs.append(float(words[0]))
                ys.append(float(words[1]))

        return np.asarray(xs), np.asarray(ys)

    def __init__(self,
        f_nodes_int, f_nodes_bdy,
        f_samples_int, f_samples_bdy,
        sample_frac=1.0):

        self.f_nodes_int = f_nodes_int
        self.f_nodes_bdy = f_nodes_bdy

        self.f_samples_int = f_samples_int
        self.f_samples_bdy = f_samples_bdy

        self.sample_frac = sample_frac

        # Interior nodes: Free
        xi, yi = self._extract_coordinates(f_nodes_int)
        self.interior_nodes = (xi, yi)

        ## Boundary nodes: Fixed
        xb, yb = self._extract_coordinates(f_nodes_bdy)
        self.boundary_nodes = (xb, yb)

        ## Interior samples
        xi, yi = self._extract_coordinates(f_samples_int)
        self.interior_samples = (tensor(xi, requires_grad=True),
                                 tensor(yi, requires_grad=True))

        self.n_interior = len(xi)
        self.rng_interior = np.arange(self.n_interior)
        self.sample_size = int(self.sample_frac*self.n_interior)

        ## Boundary samples
        xb, yb = self._extract_coordinates(f_samples_bdy)
        self.boundary_samples = (tensor(xb, requires_grad=True),
                                 tensor(yb, requires_grad=True))

    def nodes(self):
        return self.interior_nodes

    def fixed_nodes(self):
        return self.boundary_nodes

    def interior(self):
        if abs(self.sample_frac - 1.0) < 1e-3:
            return self.interior_samples
        else:
            idx = np.random.choice(self.rng_interior,
                size=self.sample_size, replace=False)
            x, y = self.interior_samples
            return x[idx], y[idx]

    def boundary(self):
        return self.boundary_samples

    def plot_points(self):
        xi, yi = self._extract_coordinates(self.f_samples_int)
        return (tensor(xi), tensor(yi))

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        return uxx + uyy + 1.0

    def has_exact(self):
        return False

    def _get_residue(self, nn):
        xs, ys = self.interior()
        u = nn(xs, ys)
        u, ux, uy, uxx, uyy = self._compute_derivatives(u, xs, ys)
        res = self.pde(xs, ys, u, ux, uy, uxx, uyy)
        return res

    def interior_loss(self, nn):
        res = self._get_residue(nn)
        return (res**2).mean()

    def boundary_loss(self, nn):
        xb, tb = self.boundary()
        u = nn(xb, tb)
        ub = 0.0
        bc = u - ub
        return (bc**2).sum()


def _vtu2data(fname):
    src = mlab.pipeline.open(fname, figure=False)
    ug = src.reader.output
    pts = ug.points.to_array()
    scalar = ug.point_data.scalars.to_array()
    return pts, scalar


def _get_errors(nn, fvtu):
    pts, u_exact = _vtu2data(fvtu)
    x = pts[:,0]
    y = pts[:,1]
    xt, yt = tensor(x.ravel()), tensor(y.ravel())
    u = nn(xt, yt).detach().cpu().numpy()
    u.shape = x.shape

    du = u - u_exact
    L1 = np.mean(np.abs(du))
    L2 = np.sqrt(np.mean(du**2))
    Linf = np.max(np.abs(du))

    return L1, L2, Linf


class PointCloud(Plotter2D):
    def get_plot_data(self):
        x, y = self.pde.plot_points()
        pn = self.nn(x, y).detach().cpu().numpy()
        pn.shape = x.shape
        xn = x.detach().cpu().numpy()
        yn = y.detach().cpu().numpy()
        return xn, yn, pn

    def plot_solution(self):
        xn, yn, pn = self.get_plot_data()
        if self.plt1 is None:
            mlab.figure(size=(700, 700), fgcolor=(0,0,0), bgcolor=(1,1,1))
            self.plt1 = mlab.points3d(xn, yn, pn, pn)
        else:
            self.plt1.mlab_source.scalars = pn
        return self.get_error(xn, yn, pn)


def plot(app):
    fvtu = 'fem/poisson_irregular000000.vtu'
    L1, L2, Linf = _get_errors(app.nn, fvtu)

    print("L1 error = ", L1)
    print("L2 error = ", L2)
    print("Linf error = ", Linf)

    mlab.figure(size=(700, 700), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    pts, u_fem = _vtu2data(fvtu)
    src = mlab.pipeline.open(fvtu)
    s = mlab.pipeline.surface(src)
    mlab.colorbar(s)
    mlab.show()


if __name__ == '__main__':
    app = App2D(
        pde_cls=Poisson2D,
        nn_cls=SPINN2D,
        plotter_cls=PointCloud
    )
    app.run(lr=1e-3)
    #plot(app)

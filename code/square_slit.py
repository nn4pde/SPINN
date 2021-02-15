import numpy as np
from twod import Problem2D, RegularDomain, App2D, tensor
from spinn2d import SPINN2D


class SlitDomain(RegularDomain):
    def __init__(self, n_nodes, ns, nb=None, nbs=None):
        # Interior nodes
        n = round(np.sqrt(n_nodes) + 0.49)
        self.n = n
        dxb2 = 1.0/(n + 1)
        xl, xr = dxb2 - 1.0, 1.0 - dxb2
        sl = slice(xl, xr, n*1j)
        x, y = np.mgrid[sl, sl]
        cond = ~((x >= 0) & (np.abs(y) < dxb2))
        self.i_nodes = (x[cond].ravel(), y[cond].ravel())

        # Fixed nodes
        nb = n if nb is None else nb
        self.nb = nb
        dxb2 = 1.0/(nb)
        _x = np.linspace(dxb2 - 1.0, 1.0 - dxb2, nb)
        _o = np.ones_like(_x)
        nslit = int(nb//2 + 1)
        x = np.hstack((_x, _o, _x, -1*_o, np.linspace(0, 1, nslit)))
        y = np.hstack((_o*-1, _x, _o, _x, np.zeros(nslit)))
        self.f_nodes = (x, y)

        # Interior samples
        self.ns = ns = round(np.sqrt(ns) + 0.49)
        dxb2 = 1.0/(ns)
        xl, xr = dxb2 - 1.0, 1.0 - dxb2
        sl = slice(xl, xr, ns*1j)
        x, y = np.mgrid[sl, sl]
        cond = ~((x >= 0) & (np.abs(y) < dxb2))
        xs, ys = (tensor(t.ravel(), requires_grad=True)
                  for t in (x[cond], y[cond]))
        self.p_samples = (xs, ys)

        # Boundary samples
        nbs = ns if nbs is None else nbs
        self.nbs = nbs
        sl = slice(-1.0, 1.0, nbs*1j)
        x, y = np.mgrid[sl, sl]
        cond = ((x < xl) | (x > xr)) | ((y < xl) | (y > xr))
        x, y = x[cond].ravel(), y[cond].ravel()
        nslit = int(nbs//2 + 1)
        xb = np.hstack((x, np.linspace(0, 1.0, nslit)))
        yb = np.hstack((y, np.zeros(nslit)))
        xb, yb = (tensor(t, requires_grad=True) for t in (xb, yb))
        self.b_samples = (xb, yb)

    def plot_points(self):
        n = self.ns*2
        x, y = np.mgrid[-1:1:n*1j, -1:1:n*1j]
        return x, y

    def eval_bc(self, problem):
        xb, yb = self.boundary()
        u = problem.nn(xb, yb)
        ub = 0.0
        return u - ub


class SlitProblem(Problem2D):
    @classmethod
    def from_args(cls, domain, nn, args):
        return cls(domain, nn, None)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        pass

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        return uxx + uyy + 1.0

    def exact(self, x, y):
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        return 0.15*np.sqrt(r)*np.sin(0.5*theta)

    def has_exact(self):
        return False


if __name__ == '__main__':
    app = App2D(
        problem_cls=SlitProblem, nn_cls=SPINN2D,
        domain_cls=SlitDomain
    )
    app.run(nodes=50, samples=200, lr=1e-2)

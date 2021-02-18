import numpy as np
from common import tensor
from pde_basic_2d import RegularPDE
from spinn2d import Problem2D, App2D, SPINN2D, tensor


class IVPDE(RegularPDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples, args.b_nodes,
                   args.b_samples, args.de, args.ic, args.viscosity)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        super().setup_argparse(parser, **kw)
        p = parser
        p.add_argument(
            '--de', dest='de', default=kw.get('de', 'linear'),
            choices=['linear', 'burgers'],
            help='Differential equation to solve.'
        )
        p.add_argument(
            '--viscosity', dest='viscosity',
            default=kw.get('viscosity', 0.0), type=float,
            help='Differential equation to solve.'
        )
        p.add_argument(
            '--ic', dest='ic', default=kw.get('ic', 'gaussian'),
            choices=['gaussian', 'hat', 'sin', 'sin2'],
            help='Initial condition.'
        )

    def __init__(self, n_nodes, ns, nb=None, nbs=None,
                 deq='linear', ic='gaussian', viscosity=0.0):
        self.deq = deq
        self.ic = ic
        self.viscosity = viscosity
        # Interior nodes
        n = round(np.sqrt(n_nodes) + 0.49)
        self.n = n
        xl, xr = -1.0, 1.0
        slx = slice(xl, xr, n*1j)
        ny = int(n//2)
        dyb2 = 0.5/ny
        sly = slice(dyb2, 1.0, ny*1j)
        x, y = np.mgrid[slx, sly]
        self.i_nodes = (x.ravel(), y.ravel())

        # Fixed nodes
        x = np.linspace(xl, xr, n*2)
        y = np.zeros_like(x)
        self.f_nodes = (x, y)
        # self.f_nodes = (np.array([]), np.array([]))

        # Interior samples
        self.ns = ns = round(np.sqrt(ns) + 0.49)
        dxb2 = 1.0/(ns)
        xl, xr = -1.0, 1.0
        slx = slice(xl, xr, ns*1j)
        ny = int(ns//2)
        slt = slice(dxb2, xr, ny*1j)
        xs, ys = (tensor(t.ravel(), requires_grad=True)
                  for t in np.mgrid[slx, slt])
        self.p_samples = (xs, ys)

        # Boundary samples (really the initial value samples)
        nbs = ns if nbs is None else nbs
        self.nbs = nbs
        x = np.linspace(-1.0, 1.0, nbs)
        y = np.zeros_like(x)
        xb, yb = (tensor(t.ravel(), requires_grad=True) for t in (x, y))
        self.b_samples = (xb, yb)

    def eval_bc(self, problem):
        xb, yb = self.boundary()
        xbn, ybn = (t.detach().cpu().numpy() for t in (xb, yb))

        u = problem.nn(xb, yb)
        ub = tensor(self.exact(xbn, ybn))
        return u - ub

    def plot_points(self):
        n = self.ns*2
        x, y = np.mgrid[-1:1:n*1j, 0:1:n*1j]
        return x, y

    def pde(self, x, y, u, ux, uy, uxx, uyy):
        if self.deq == 'linear':
            a = 0.5
            return uy + a*ux
        elif self.deq == 'burgers':
            return uy + u*ux - self.viscosity*uxx

    def exact(self, x, y):
        a = 0.5
        if self.ic == 'hat':
            x1 = (x - a*y + 0.35)
            return np.heaviside(x1, 0.5) - np.heaviside(x1 - 0.5, 0.5)
        elif self.ic == 'gaussian':
            x1 = (x - a*y + 0.3)/0.15
            return np.exp(-x1**2)
        elif self.ic == 'sin':
            x1 = (x - a*y + 0.5)
            y = np.heaviside(x1, 0.5) - np.heaviside(x1 - 0.5, 0.5)
            return np.sin(x1*np.pi*2*y)
        elif self.ic == 'sin2':
            x1 = (x - a*y + 0.5)
            y = np.heaviside(x1, 0.5) - np.heaviside(x1 - 0.5, 0.5)
            return np.sin(x1*np.pi*4*y)


if __name__ == '__main__':
    app = App2D(
        problem_cls=Problem2D, nn_cls=SPINN2D,
        pde_cls=IVPDE
    )
    app.run(nodes=100, samples=200, lr=1e-2)

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from mayavi import mlab
import numpy as np
import torch
import torch.autograd as ag
import torch.nn as nn

from oned import Case1D, Solver, DiffEq


PI = np.pi


class Case2D(Case1D):
    @classmethod
    def from_args(cls, nn, args):
        return cls(nn, args.de, args.samples)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        # Case options
        p.add_argument(
            '--samples', '-s', dest='samples',
            default=kw.get('samples', 100), type=int,
            help='Number of sample points to use.'
        )

    def __init__(self, nn, deq, n_samples):
        '''Initializer

        Parameters
        -----------

        nn: Neural network for the solution
        eq: DiffEq: Differential equation to evaluate.
        n_samples: int: number of sample points.
        '''
        self.nn = nn
        self.deq = deq
        self.plt1 = None
        self.plt2 = None  # For weights
        self.ns = ns = round(np.sqrt(n_samples) + 0.49)
        dxb2 = 0.5/(ns)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, ns*1j)
        xs, ys = (self._get_array(t) for t in np.mgrid[sl, sl])
        self.points = (xs, ys)
        n = ns + 2
        x, y = np.mgrid[0:1:n*1j, 0:1:n*1j]
        cond = ((x < xl) | (x > xr)) | ((y < xl) | (y > xr))
        xb, yb = (self._get_array(t) for t in (x[cond], y[cond]))
        self.points_b = (xb, yb)
        self.ub_ex = self.deq.exact(xb, yb)
        for t in (xs, ys, xb, yb):
            t.requires_grad = True

    def _get_array(self, x):
        return torch.tensor(x.ravel()).float().unsqueeze(1)

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

    def loss(self):
        nn = self.nn
        xs, ys = self.points
        u = nn(xs, ys)
        u, ux, uy, uxx, uyy = self._compute_derivatives(u, xs, ys)
        res = self.deq.ode(xs, ys, u, ux, uy, uxx, uyy)
        ub = nn(*self.points_b)
        ub_ex = self.ub_ex
        l1 = (res**2).mean()
        l2 = ((ub - ub_ex)**2).sum()
        loss = l1 + l2
        return loss

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        nn = self.nn
        x = nn.layer1.x.detach().numpy()
        y = nn.layer1.y.detach().numpy()
        if not self.plt2:
            self.plt2 = mlab.points3d(
                x, y, np.zeros_like(x), scale_factor=0.025
            )
        else:
            self.plt2.mlab_source.trait_set(x=x, y=y)

    def get_plot_data(self):
        n = 2*self.ns
        x, y = (self._get_array(t) for t in np.mgrid[0:1:n*1j, 0:1:n*1j])
        pn = self.nn(x, y).detach().squeeze().numpy()
        xn = x.squeeze().numpy()
        yn = y.squeeze().numpy()
        yn.shape = xn.shape = pn.shape = (n, n)
        return xn, yn, pn

    def plot_solution(self):
        xn, yn, pn = self.get_plot_data()
        if self.plt1 is None:
            un = self.deq.exact(xn, yn)
            mlab.figure(size=(700, 700))
            mlab.surf(xn, yn, un, representation='wireframe')
            self.plt1 = mlab.surf(xn, yn, pn, opacity=0.8)
            mlab.colorbar()
        else:
            self.plt1.mlab_source.scalars = pn
        return self.get_error(xn, yn, pn)

    def plot(self):
        err = self.plot_solution()
        self.plot_weights()
        mlab.process_ui_events()
        self.plt1.scene._lift()
        return err

    def show(self):
        mlab.show()

    def get_error(self, xn=None, yn=None, pn=None):
        if xn is None and pn is None:
            xn, yn, pn = self.get_plot_data()
        un = self.deq.exact(xn, yn)
        umax = np.max(np.abs(un))
        diff = np.abs(un - pn)
        return diff.mean()/umax

    def save(self, dirname):
        '''Save the model and results.

        '''
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(self.nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz')
        x, y, u = self.get_plot_data()
        u_exact = self.deq.exact(x, y)
        np.savez(rfile, x=x, y=y, u=u, u_exact=u_exact)


class Shift2D(nn.Module):
    def __init__(self, points, fixed_h=False):
        super().__init__()
        x = points[0].ravel()
        y = points[1].ravel()
        self.n = x.size
        dx = np.ptp(x)/np.sqrt(self.n)
        self.dx = dx
        self.x = nn.Parameter(torch.tensor(x).float())
        self.y = nn.Parameter(torch.tensor(y).float())

        if fixed_h:
            self.h = nn.Parameter(torch.tensor(dx))
        else:
            self.h = nn.Parameter(dx*torch.ones_like(self.x))

    def forward(self, x, y):
        fac = 1.0/self.h
        xh = (x - self.x)*fac
        yh = (y - self.y)*fac
        return xh, yh


class PoissonSin2(DiffEq):
    def ode(self, x, y, u, ux, uy, uxx, uyy):
        f = 20*PI**2*torch.sin(2*PI*x)*torch.sin(4*PI*y)
        return uxx + uyy + f

    def exact(self, x, y):
        return np.sin(2*PI*x)*np.sin(4*PI*y)


class PoissonBump(DiffEq):
    def ode(self, x, y, u, ux, uy, uxx, uyy):
        K = 0.05
        ex = torch.exp(-(x - 0.25)*(x - 0.25)/K)
        fxx = (
            (1.0 + ((1.0 - 2.0*x)*(x - 0.25) + x*(1.0 - x))/K)
            + ((1.0 - 2.0*x - 2.0*x*(1 - x)*(x - 0.25)/K)*(x - 0.25)/K)
        )*2.0*ex*y*(1 - y)
        fyy = 2.0*x*(1.0 - x)*ex
        return uxx + uyy + fxx + fyy

    def exact(self, x, y):
        K = 0.05
        return x*(1.0 - x)*y*(1.0 - y)*np.exp(-(x - 0.25)*(x - 0.25)/K)


def gaussian(x, y):
    return torch.exp(-(x*x + y*y))


class SoftPlus:
    def __init__(self):
        self._sp = nn.Softplus()
        self.k = torch.Tensor([1.0 + 4.0*np.log(2.0)])
        self.fac = self._sp(torch.Tensor([1.0]))

    def __call__(self, x, y):
        sp = self._sp
        return sp(
            self.k - sp(x) - sp(-x) - sp(y) - sp(-y)
        )*self.fac


class Kernel(nn.Module):
    def __init__(self, n_kernel, activation=torch.tanh):
        super().__init__()

        self.activation = activation
        self.n_kernel = n_kernel
        self.layer1 = nn.Linear(1, n_kernel)
        self.layer2 = nn.Linear(n_kernel, 2*n_kernel)
        self.layer3 = nn.Linear(2*n_kernel, n_kernel)
        self.layer4 = nn.Linear(n_kernel, 1)

    def forward(self, x, y):
        r = x*x + y*y
        act = self.activation
        orig_shape = r.shape
        r = r.flatten().unsqueeze(1)
        r = act(self.layer1(r))
        r = act(self.layer2(r))
        r = act(self.layer3(r))
        r = self.layer4(r)
        r = r.reshape(orig_shape)
        return r


def setup_argparse(*cls, **kw):
    '''Setup the argument parser.

    Any keyword arguments are used as the default values for the parameters.
    '''
    p = ArgumentParser(
        description="Configurable options.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    for c in cls:
        c.setup_argparse(p, **kw)
    # Differential equation to solve.
    p.add_argument(
        '--de', dest='de', default=kw.get('de', 'sin2'),
        choices=['sin2', 'bump'],
        help='Differential equation to solve.'
    )
    p.add_argument(
        '--activation', '-a', dest='activation',
        default=kw.get('activation', 'gaussian'),
        choices=['gaussian', 'softplus', 'kernel'],
        help='Select the activation function for particles.'
    )
    p.add_argument(
        '--kernel-size', dest='kernel_size',
        default=kw.get('kernel_size', 10), type=int,
        help='Activation kernel size (in place of a Gaussian).'
    )

    return p


def update_args(args):
    '''Updates objects in args as per options selected

    This will change the args so that the values are appropriate objects and
    not strings.

    '''

    activations = {
        'gaussian': lambda x: gaussian,
        'softplus': lambda x: SoftPlus(),
        'kernel': Kernel
    }
    args.activation = activations[args.activation](args.kernel_size)

    des = {
        'sin2': PoissonSin2, 'bump': PoissonBump
    }
    args.de = des[args.de]()


def main(nn_cls, case_cls, solver=Solver, **kw):
    parser = setup_argparse(solver, nn_cls, case_cls, **kw)
    args = parser.parse_args()
    update_args(args)

    nn = nn_cls.from_args(args)
    case = case_cls.from_args(nn, args)
    solver = Solver.from_args(case, args)
    solver.solve()

    if args.directory is not None:
        solver.save(args.directory)

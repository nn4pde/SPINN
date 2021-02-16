import os
import sys
from mayavi import mlab
import numpy as np
import torch
import torch.autograd as ag
import torch.nn as nn

from common import PDE, tensor
from spinn1d import Problem1D, App1D


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


class ToyPDE(RegularPDE):
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
            return uxx + uyy + f
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


class Problem2D(Problem1D):
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
        pde = self.pde
        nn = self.nn
        xs, ys = pde.interior()
        u = nn(xs, ys)
        u, ux, uy, uxx, uyy = self._compute_derivatives(u, xs, ys)
        res = pde.pde(xs, ys, u, ux, uy, uxx, uyy)
        l1 = (res**2).mean()
        bc = pde.eval_bc(self)
        bc_loss = (bc**2).sum()
        loss = l1 + bc_loss
        return loss

    def get_error(self, xn=None, yn=None, pn=None):
        if not self.pde.has_exact():
            return 0.0
        if xn is None and pn is None:
            xn, yn, pn = self.get_plot_data()
        un = self.pde.exact(xn, yn)
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
        u_exact = self.exact(x, y)
        np.savez(rfile, x=x, y=y, u=u, u_exact=u_exact)

    # Plotting methods
    def get_plot_data(self):
        x, y = self.pde.plot_points()
        xt, yt = tensor(x.ravel()), tensor(y.ravel())
        pn = self.nn(xt, yt).detach().cpu().numpy()
        pn.shape = x.shape
        return x, y, pn

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        nn = self.nn
        x, y = nn.centers()
        h = nn.widths().detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        if not self.plt2:
            self.plt2 = mlab.points3d(
                x, y, np.zeros_like(x), h, mode='2dcircle',
                scale_factor=1.0
            )
            self.plt2.glyph.glyph_source.glyph_source.resolution = 20
        else:
            self.plt2.mlab_source.trait_set(x=x, y=y, scalars=h)

    def plot_solution(self):
        xn, yn, pn = self.get_plot_data()
        pde = self.pde
        if self.plt1 is None:
            mlab.figure(size=(700, 700))
            if self.show_exact and pde.has_exact():
                un = pde.exact(xn, yn)
                mlab.surf(xn, yn, un, representation='wireframe')
            self.plt1 = mlab.surf(xn, yn, pn, opacity=0.8)
            mlab.colorbar(self.plt1)
        else:
            self.plt1.mlab_source.scalars = pn
        return self.get_error(xn, yn, pn)

    def plot(self):
        err = self.plot_solution()
        self.plot_weights()
        mlab.process_ui_events()
        if sys.platform.startswith('linux'):
            self.plt1.scene._lift()
        return err

    def show(self):
        mlab.show()


class Shift2D(nn.Module):
    def __init__(self, points, fixed_points, fixed_h=False):
        super().__init__()
        n_free = len(points[0])
        n_fixed = len(fixed_points[0])
        self.n = n = n_free + n_fixed
        if n_fixed:
            dsize = np.ptp(np.hstack((points[0], fixed_points[0])))
        else:
            dsize = np.ptp(points[0])
        dx = dsize/np.sqrt(n)
        self.dx = dx
        self.x = nn.Parameter(tensor(points[0]))
        self.y = nn.Parameter(tensor(points[1]))
        fp = fixed_points
        self.xf = tensor(fp[0])
        self.yf = tensor(fp[1])

        self.fixed_h = fixed_h
        if fixed_h:
            self.h = nn.Parameter(tensor(dx))
        else:
            self.h = nn.Parameter(dx*torch.ones(n))

    def centers(self):
        return torch.cat((self.x, self.xf)), torch.cat((self.y, self.yf))

    def widths(self):
        if self.fixed_h:
            return self.h*torch.ones(self.n)
        else:
            return self.h

    def forward(self, x, y):
        xc, yc = self.centers()
        fac = 1.0/self.h
        xh = (x - xc)*fac
        yh = (y - yc)*fac
        return xh, yh


class SPINN2D(nn.Module):
    @classmethod
    def from_args(cls, pde, activation, args):
        return cls(pde, activation,
                   fixed_h=args.fixed_h, use_pu=not args.no_pu)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--fixed-h', dest='fixed_h', action='store_true', default=False,
            help='Use fixed width nodes.'
        )
        p.add_argument(
            '--no-pu', dest='no_pu', action='store_true', default=False,
            help='Do not use a partition of unity.'
        )

    def __init__(self, pde, activation, n_outputs=1,
                 fixed_h=False, use_pu=True):
        super().__init__()

        self.activation = activation
        self.use_pu = use_pu
        self.layer1 = Shift2D(pde.nodes(), pde.fixed_nodes(),
                              fixed_h=fixed_h)
        n = self.layer1.n
        self.layer2 = nn.Linear(n, n_outputs, bias=not use_pu)
        self.layer2.weight.data.fill_(0.0)
        if not self.use_pu:
            self.layer2.bias.data.fill_(0.0)

    def forward(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        xh, yh = self.layer1(x, y)
        z = self.activation(xh, yh)
        if self.use_pu:
            zsum = z.sum(axis=1).unsqueeze(1)
        else:
            zsum = 1.0
        z = self.layer2(z/zsum)
        return z.squeeze()

    def centers(self):
        return self.layer1.centers()

    def widths(self):
        return self.layer1.widths()

    def weights(self):
        return self.layer2.weight


def gaussian(x, y):
    return torch.exp(-(x*x + y*y))


class SoftPlus:
    def __init__(self):
        self._sp = nn.Softplus()
        self.k = tensor([1.0 + 4.0*np.log(2.0)])
        self.fac = self._sp(tensor([1.0]))

    def __call__(self, x, y):
        sp = self._sp
        return sp(
            self.k - sp(x) - sp(-x) - sp(y) - sp(-y)
        )/self.fac


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


class App2D(App1D):

    def _setup_activation_options(self, p, **kw):
        # Differential equation to solve.
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

    def _get_activation(self, args):
        activations = {
            'gaussian': lambda x: gaussian,
            'softplus': lambda x: SoftPlus(),
            'kernel': Kernel
        }
        return activations[args.activation](args.kernel_size)


if __name__ == '__main__':
    app = App2D(
        problem_cls=Problem2D, nn_cls=SPINN2D,
        pde_cls=ToyPDE
    )
    app.run(nodes=40, samples=120, lr=1e-2)

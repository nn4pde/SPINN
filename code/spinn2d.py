import os
import sys
from mayavi import mlab
import numpy as np
import torch
import torch.nn as nn

from common import App, Plotter, tensor


class Plotter2D(Plotter):
    def get_error(self, xn=None, yn=None, pn=None):
        if not self.pde.has_exact():
            return 0.0, 0.0, 0.0

        if xn is None and pn is None:
            xn, yn, pn = self.get_plot_data()

        un = self.pde.exact(xn, yn)
        diff = un - pn
        err_L1 = np.mean(np.abs(diff))
        err_L2 = np.sqrt(np.mean(diff**2))
        err_Linf = max(np.abs(diff))
        return err_L1, err_L2, err_Linf

    def save(self, dirname):
        '''Save the model and results.

        '''
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(self.nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz')
        x, y, u = self.get_plot_data()
        u_exact = self.pde.exact(x, y)
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
                   fixed_h=args.fixed_h, use_pu=args.pu)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--fixed-h', dest='fixed_h', action='store_true', default=False,
            help='Use fixed width nodes.'
        )
        p.add_argument(
            '--pu', dest='pu', action='store_true', default=False,
            help='Use a partition of unity.'
        )

    def __init__(self, pde, activation, fixed_h=False, use_pu=False):
        super().__init__()

        self.activation = activation
        self.use_pu = use_pu
        self.layer1 = Shift2D(pde.nodes(), pde.fixed_nodes(),
                              fixed_h=fixed_h)
        n = self.layer1.n
        self.layer2 = nn.Linear(n, pde.n_vars(), bias=not use_pu)
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
    return torch.exp(-0.5*(x*x + y*y))


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


class App2D(App):

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

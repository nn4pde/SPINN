import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import Plotter, App, tensor


class Plotter1D(Plotter):
    def get_error(self, xn=None, pn=None):
        if not self.pde.has_exact():
            return 0.0

        if xn is None and pn is None:
            xn, pn = self.get_plot_data()
        yn = self.pde.exact(xn)
        umax = max(np.abs(yn))
        diff = np.abs(yn - pn)
        return diff.mean()/umax

    def save(self, dirname):
        '''Save the model and results.

        '''
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(self.nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz')
        x, y = self.get_plot_data()
        y_exact = self.pde.exact(x)
        np.savez(rfile, x=x, y=y, y_exact=y_exact)

    # Plotting methods
    def get_plot_data(self):
        x = self.pde.plot_points()
        xt = tensor(x)
        pn = self.nn(xt).detach().cpu().numpy()
        return x, pn

    def plot_solution(self):
        xn, pn = self.get_plot_data()
        pde = self.pde
        if self.plt1 is None:
            lines, = plt.plot(xn, pn, '-', label='computed')
            self.plt1 = lines
            if self.show_exact and pde.has_exact():
                yn = self.pde.exact(xn)
                plt.plot(xn, yn, label='exact')
            else:
                yn = pn
            plt.grid()
            plt.xlim(-0.1, 1.1)
            ymax, ymin = np.max(yn), np.min(yn)
            delta = (ymax - ymin)*0.5
            plt.ylim(ymin - delta, ymax + delta)
        else:
            self.plt1.set_data(xn, pn)
        return self.get_error(xn, pn)

    def plot_weights(self):
        x = self.nn.centers().detach().cpu().numpy()
        w = self.nn.weights().detach().cpu().squeeze().numpy()
        if not self.plt2:
            self.plt2, = plt.plot(x, np.zeros_like(x), 'o', label='centers')
            self.plt3, = plt.plot(x, w, 'o', label='weights')
        else:
            self.plt2.set_data(x, np.zeros_like(x))
            self.plt3.set_data(x, w)

    def plot(self):
        err = self.plot_solution()
        self.plot_weights()
        plt.legend()
        plt.pause(0.01)
        return err

    def show(self):
        plt.show()


class Shift(nn.Module):
    def __init__(self, points, fixed_points, fixed_h=False):
        super().__init__()
        n_free = len(points)
        n_fixed = len(fixed_points)
        self.n = n = n_free + n_fixed
        if n_fixed:
            xmin = min(np.min(points), np.min(fixed_points))
            xmax = max(np.max(points), np.max(fixed_points))
        else:
            xmin, xmax = np.min(points), np.max(points)
        dx = (xmax - xmin)/n
        self.center = nn.Parameter(tensor(points))
        self.fixed = tensor(fixed_points)
        if fixed_h:
            self.h = nn.Parameter(torch.tensor(dx))
        else:
            self.h = nn.Parameter(dx*torch.ones(n))

    def centers(self):
        return torch.cat((self.center, self.fixed))

    def forward(self, x):
        cen = self.centers()
        return (x - cen)/self.h


class SPINN1D(nn.Module):
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

        self.fixed_h = fixed_h
        self.use_pu = use_pu
        self.layer1 = Shift(pde.nodes(), pde.fixed_nodes(),
                            fixed_h=fixed_h)
        n = self.layer1.n
        self.activation = activation
        self.layer2 = nn.Linear(n, pde.n_vars(), bias=not use_pu)
        self.layer2.weight.data.fill_(0.0)
        if not self.use_pu:
            self.layer2.bias.data.fill_(0.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.layer1(x)
        y = self.activation(y)
        if self.use_pu:
            y1 = y.sum(axis=1).unsqueeze(1)
        else:
            y1 = tensor(1.0)
        y = self.layer2(y/y1)
        return y.squeeze()

    def centers(self):
        return self.layer1.centers()

    def widths(self):
        return self.layer1.h

    def weights(self):
        return self.layer2.weight

    def show(self):
        print("Basis centers: ", self.centers())
        print("Mesh widths: ", self.widths())
        print("Nodal weights: ", self.weights())
        print("Output bias: ", self.layer2.bias)


# Activation functions

def gaussian(x):
    return torch.exp(-0.5*x*x)


class SoftPlus:
    def __init__(self):
        self._sp = nn.Softplus()
        self.k = tensor([1.0 + 2.0*np.log(2.0)])
        self.fac = self._sp(tensor([1.0]))

    def __call__(self, x):
        sp = self._sp
        return sp(self.k - sp(x) - sp(-x))/self.fac


tanh = torch.tanh


class Kernel(nn.Module):
    def __init__(self, n_kernel):
        super().__init__()

        self.n_kernel = n_kernel
        self.layer1 = nn.Linear(1, n_kernel)
        self.layer2 = nn.Linear(n_kernel, n_kernel)
        self.layer3 = nn.Linear(n_kernel, 1)

    def forward(self, x):
        orig_shape = x.shape
        x = x.flatten().unsqueeze(1)
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        x = x.reshape(orig_shape)
        return x


class App1D(App):
    def _setup_activation_options(self, p, **kw):
        p.add_argument(
            '--activation', '-a', dest='activation',
            default=kw.get('activation', 'gaussian'),
            choices=['gaussian', 'softplus', 'tanh', 'kernel'],
            help='Select the activation function for particles.'
        )
        p.add_argument(
            '--kernel-size', dest='kernel_size',
            default=kw.get('kernel_size', 5), type=int,
            help='Activation kernel size (in place of a Gaussian).'
        )

    def _get_activation(self, args):
        activations = {
            'gaussian': lambda x: gaussian,
            'tanh': lambda x: tanh,
            'softplus': lambda x: SoftPlus(),
            'kernel': Kernel
        }
        return activations[args.activation](args.kernel_size)

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as ag
import torch.nn as nn

from common import Problem, Optimizer, DiffEq, device, tensor


class Problem1D(Problem):

    @classmethod
    def from_args(cls, nn, args):
        return cls(nn, args.de, args.samples)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        # Problem options
        p.add_argument(
            '--samples', '-s', dest='samples',
            default=kw.get('samples', 30), type=int,
            help='Number of sample points to use.'
        )

    def __init__(self, nn, deq, ns):
        '''Initializer

        Parameters
        -----------

        nn: Neural network for the solution
        eq: DiffEq: Differential equation to evaluate.
        ns: int: number of sample points.
        '''
        self.nn = nn
        self.deq = deq
        self.plt1 = None
        self.plt2 = None  # For weights
        self.ns = ns
        self.xs = tensor(np.linspace(0, 1, ns), requires_grad=True)

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

    def loss(self):
        nn = self.nn
        xs = self.xs
        u = nn(xs)
        u, ux, uxx = self._compute_derivatives(u, xs)
        res = self.deq.ode(xs, u, ux, uxx)
        ub = u[[0, -1]]
        ub_ex = tensor(self.deq.exact(torch.Tensor([0.0, 1.0])))
        ns = len(xs)

        loss = (
            (res**2).sum()/ns
            + ((ub - ub_ex)**2).sum()*ns
        )
        return loss

    def get_plot_data(self):
        n = self.ns
        x = tensor(np.linspace(0.0, 1.0, n))
        pn = self.nn(x).detach().cpu().numpy()
        xn = x.cpu().numpy()
        return xn, pn

    def plot_solution(self):
        xn, pn = self.get_plot_data()
        if self.plt1 is None:
            yn = self.deq.exact(xn)
            lines, = plt.plot(xn, pn, '-', label='computed')
            self.plt1 = lines
            plt.plot(xn, self.deq.exact(xn), label='exact')
            plt.grid()
            plt.xlim(-0.1, 1.1)
            ymax, ymin = np.max(yn), np.min(yn)
            delta = (ymax - ymin)*0.5
            plt.ylim(ymin - delta, ymax + delta)
        else:
            self.plt1.set_data(xn, pn)
        return self.get_error(xn, pn)

    def plot(self):
        err = self.plot_solution()
        self.plot_weights()
        plt.legend()
        plt.pause(0.01)
        return err

    def show(self):
        plt.show()

    def get_error(self, xn=None, pn=None):
        if xn is None and pn is None:
            xn, pn = self.get_plot_data()
        yn = self.deq.exact(xn)
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
        y_exact = self.deq.exact(x)
        np.savez(rfile, x=x, y=y, y_exact=y_exact)


class Pulse1D(DiffEq):
    def __init__(self, K=0.01):
        self.K = K

    def ode(self, x, u, ux, uxx):
        K = self.K
        f = -(K*(-6*x + 4/3.) +
              x*(2*x - 2./3)**2)*torch.exp(-(x - 1/3.)**2/K)/K**2
        return uxx + f

    def exact(self, *args):
        x = args[0]
        K = self.K
        return x*(np.exp(-((x - (1.0/3.0))**2)/K) - np.exp(-4.0/(9.0*K)))


class Sin8(DiffEq):
    def ode(self, x, u, ux, uxx):
        return uxx + torch.sin(8.0*np.pi*x)

    def exact(self, *args):
        x = args[0]
        return np.sin(8.0*np.pi*x)/(64.0*np.pi**2)


class Simple(DiffEq):
    def ode(self, x, u, ux, uxx):
        return uxx + 1.0

    def exact(self, *args):
        x = args[0]
        return 0.5*x*(1.0 - x)


# Activation functions

def gaussian(x):
    return torch.exp(-x*x)


class SoftPlus:
    def __init__(self):
        self._sp = nn.Softplus()
        self.k = tensor([1.0 + 2.0*np.log(2.0)])
        self.fac = self._sp(tensor([1.0]))

    def __call__(self, x):
        sp = self._sp
        return sp(self.k - sp(x) - sp(-x))*self.fac


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


def setup_argparse(*cls, **kw):
    '''Setup the argument parser.

    Any keyword arguments are used as the default values for the parameters.
    '''
    p = ArgumentParser(
        description="Configurable options.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        '--gpu', dest='gpu', action='store_true',
        default=kw.get('gpu', False),
        help='Run code on the GPU.'
    )

    for c in cls:
        c.setup_argparse(p, **kw)

    # Differential equation to solve.
    p.add_argument(
        '--de', dest='de', default=kw.get('de', 'sin8'),
        choices=['sin8', 'pulse', 'simple'],
        help='Differential equation to solve.'
    )
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

    return p


def update_args(args):
    '''Updates objects in args as per options selected

    This will change the args so that the values are appropriate objects and
    not strings.

    '''
    if args.gpu:
        device("cuda")
    else:
        device("cpu")

    activations = {
        'gaussian': lambda x: gaussian,
        'tanh': lambda x: tanh,
        'softplus': lambda x: SoftPlus(),
        'kernel': Kernel
    }
    args.activation = activations[args.activation](args.kernel_size)

    des = {
        'sin8': Sin8, 'simple': Simple, 'pulse': Pulse1D
    }
    args.de = des[args.de]()


def main(nn_cls, problem_cls, optimizer=Optimizer, **kw):
    parser = setup_argparse(optimizer, nn_cls, problem_cls, **kw)
    args = parser.parse_args()
    update_args(args)

    dev = device()
    nn = nn_cls.from_args(args).to(dev)
    p = problem_cls.from_args(nn, args)
    solver = optimizer.from_args(p, args)
    solver.solve()

    if args.directory is not None:
        solver.save(args.directory)

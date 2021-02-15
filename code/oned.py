from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as ag
import torch.nn as nn

from common import Problem, Optimizer, Domain, device, tensor


class RegularDomain(Domain):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples)

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

    def __init__(self, n, ns):
        self.n = n
        self.ns = ns
        dxb2 = 0.5/(n + 1)
        self.xn = np.linspace(dxb2, 1.0 - dxb2, n)
        dxb2 = 0.5/(ns + 1)
        self.xs = tensor(np.linspace(dxb2, 1-dxb2, ns), requires_grad=True)
        self.xbn = np.array([0.0, 1.0])
        self.xb = tensor(self.xbn)

    def nodes(self):
        return self.xn

    def fixed_nodes(self):
        return self.xbn

    def interior(self):
        return self.xs

    def boundary(self):
        return self.xb

    def plot_points(self):
        n = self.ns*2
        x = np.linspace(0.0, 1.0, n)
        return x

    def eval_bc(self, problem):
        u = problem.nn(self.boundary())
        ub = tensor(problem.exact(self.xbn))
        return u - ub


class Problem1D(Problem):

    @classmethod
    def from_args(cls, domain, nn, args):
        return cls(domain, nn, args.de)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--de', dest='de', default=kw.get('de', 'sin8'),
            choices=['sin8', 'pulse', 'simple'],
            help='Differential equation to solve.'
        )

    def __init__(self, domain, nn, deq):
        '''Initializer

        Parameters
        -----------

        domain: Domain: object managing the domain.
        nn: Neural network for the solution
        eq: DiffEq: Differential equation to evaluate.
        '''
        self.domain = domain
        self.nn = nn
        self.deq = deq
        self.plt1 = None
        self.plt2 = None  # For weights

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
        domain = self.domain
        nn = self.nn
        xs = domain.interior()
        u = nn(xs)
        u, ux, uxx = self._compute_derivatives(u, xs)
        res = self.pde(xs, u, ux, uxx)
        bc = domain.eval_bc(self)
        bc_loss = (bc**2).sum()
        ns = len(xs)
        loss = (
            (res**2).mean()
            + bc_loss*ns
        )
        return loss

    def get_error(self, xn=None, pn=None):
        if not self.has_exact():
            return 0.0

        if xn is None and pn is None:
            xn, pn = self.get_plot_data()
        yn = self.exact(xn)
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
        y_exact = self.exact(x)
        np.savez(rfile, x=x, y=y, y_exact=y_exact)

    # Plotting methods
    def get_plot_data(self):
        x = self.domain.plot_points()
        xt = tensor(x)
        pn = self.nn(xt).detach().cpu().numpy()
        return x, pn

    def plot_solution(self):
        xn, pn = self.get_plot_data()
        if self.plt1 is None:
            yn = self.exact(xn)
            lines, = plt.plot(xn, pn, '-', label='computed')
            self.plt1 = lines
            if self.has_exact():
                plt.plot(xn, yn, label='exact')
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

    # ####### Override these for different differential equations
    def pde(self, x, u, ux, uxx):
        if self.deq == 'pulse':
            K = 0.01
            f = -(K*(-6*x + 4/3.) +
                  x*(2*x - 2./3)**2)*torch.exp(-(x - 1/3.)**2/K)/K**2
            return uxx + f
        elif self.deq == 'sin8':
            return uxx + torch.sin(8.0*np.pi*x)
        elif self.deq == 'simple':
            return uxx + 1.0

    def exact(self, x):
        if self.deq == 'pulse':
            K = 0.01
            return x*(np.exp(-((x - (1.0/3.0))**2)/K) -
                      np.exp(-4.0/(9.0*K)))
        elif self.deq == 'sin8':
            return np.sin(8.0*np.pi*x)/(64.0*np.pi**2)
        elif self.deq == 'simple':
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


class App1D:
    def __init__(self, problem_cls, nn_cls, domain_cls,
                 optimizer=Optimizer):
        self.problem_cls = problem_cls
        self.nn_cls = nn_cls
        self.domain_cls = domain_cls
        self.optimizer = optimizer

    def setup_argparse(self, **kw):
        '''Setup the argument parser.

        Any keyword arguments are used as the default values for the
        parameters.
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
        classes = (
            self.domain_cls, self.problem_cls, self.nn_cls, self.optimizer
        )
        for c in classes:
            c.setup_argparse(p, **kw)

        self._setup_activation_options(p)
        return p

    def _setup_activation_options(self, p, **kw):
        # Differential equation to solve.
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

    def run(self, args=None, **kw):
        parser = self.setup_argparse(**kw)
        args = parser.parse_args(args)

        if args.gpu:
            device("cuda")
        else:
            device("cpu")

        activation = self._get_activation(args)

        dev = device()
        dom = self.domain_cls.from_args(args)
        self.domain = dom
        nn = self.nn_cls.from_args(dom, activation, args).to(dev)
        self.nn = nn
        p = self.problem_cls.from_args(dom, nn, args)
        self.problem = p
        solver = self.optimizer.from_args(p, args)
        self.solver = solver
        solver.solve()

        if args.directory is not None:
            solver.save()

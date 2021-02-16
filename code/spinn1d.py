from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as ag
import torch.nn as nn

from common import Problem, Optimizer, PDE, device, tensor


class RegularPDE(PDE):
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
        ub = tensor(self.exact(self.xbn))
        return u - ub

    # ####### Override these for different differential equations
    def pde(self, x, u, ux, uxx):
        raise NotImplementedError()

    def exact(self, x, y):
        pass


class ToyPDE(RegularPDE):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples, args.de)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        super().setup_argparse(parser, **kw)
        p = parser
        p.add_argument(
            '--de', dest='de', default=kw.get('de', 'sin8'),
            choices=['sin8', 'pulse', 'simple'],
            help='Differential equation to solve.'
        )

    def __init__(self, n, ns, deq):
        super().__init__(n, ns)
        self.deq = deq

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


class Problem1D(Problem):

    @classmethod
    def from_args(cls, pde, nn, args):
        return cls(pde, nn, args.no_show_exact)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--no-show-exact', dest='no_show_exact',
            action='store_true', default=False,
            help='Do not show exact solution even if available.'
        )

    def __init__(self, pde, nn, no_show_exact=False):
        '''Initializer

        Parameters
        -----------

        pde: PDE: object managing the pde.
        nn: Neural network for the solution
        eq: DiffEq: Differential equation to evaluate.
        no_show_exact: bool: Show exact solution or not.
        '''
        self.pde = pde
        self.nn = nn
        self.plt1 = None
        self.plt2 = None  # For weights
        self.show_exact = not no_show_exact

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
        pde = self.pde
        nn = self.nn
        xs = pde.interior()
        u = nn(xs)
        u, ux, uxx = self._compute_derivatives(u, xs)
        res = pde.pde(xs, u, ux, uxx)
        bc = pde.eval_bc(self)
        bc_loss = (bc**2).sum()
        ns = len(xs)
        loss = (
            (res**2).mean()
            + bc_loss*ns
        )
        return loss

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
            self.h = nn.Parameter(torch.tensor(2.0*dx))
        else:
            self.h = nn.Parameter(2.0*dx*torch.ones(n))

    def centers(self):
        return torch.cat((self.center, self.fixed))

    def forward(self, x):
        cen = self.centers()
        return (x - cen)/self.h


class SPINN1D(nn.Module):
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

    def __init__(self, pde, activation, fixed_h=False, use_pu=True):
        super().__init__()

        self.fixed_h = fixed_h
        self.use_pu = use_pu
        self.layer1 = Shift(pde.nodes(), pde.fixed_nodes(),
                            fixed_h=fixed_h)
        n = self.layer1.n
        self.activation = activation
        self.layer2 = nn.Linear(n, 1, bias=not use_pu)
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
    def __init__(self, problem_cls, nn_cls, pde_cls,
                 optimizer=Optimizer):
        self.problem_cls = problem_cls
        self.nn_cls = nn_cls
        self.pde_cls = pde_cls
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
            self.pde_cls, self.problem_cls, self.nn_cls, self.optimizer
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
        pde = self.pde_cls.from_args(args)
        self.pde = pde
        nn = self.nn_cls.from_args(pde, activation, args).to(dev)
        self.nn = nn
        p = self.problem_cls.from_args(pde, nn, args)
        self.problem = p
        solver = self.optimizer.from_args(p, args)
        self.solver = solver
        solver.solve()

        if args.directory is not None:
            solver.save()


if __name__ == '__main__':
    app = App1D(
        problem_cls=Problem1D, nn_cls=SPINN1D,
        pde_cls=ToyPDE
    )
    app.run(nodes=20, samples=80, lr=1e-2)

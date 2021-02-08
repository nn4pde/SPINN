from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim


class Case1D:

    @classmethod
    def from_args(cls, nn, args):
        return cls(nn, args.de, args.samples)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        # Case options
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
        self.xs = torch.linspace(0, 1, ns, requires_grad=True)

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
        ub_ex = self.deq.exact(torch.Tensor([0.0, 1.0]))
        ns = len(xs)

        loss = (
            (res**2).sum()/ns
            + ((ub - ub_ex)**2).sum()*ns
        )
        return loss

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        pass

    def get_plot_data(self):
        n = self.ns
        x = torch.linspace(0.0, 1.0, n)
        pn = self.nn(x).detach().squeeze().numpy()
        xn = x.squeeze().numpy()
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


class Solver:
    @classmethod
    def from_args(cls, case, args):
        optimizers = {'Adam': optim.Adam, 'LBFGS': optim.LBFGS}
        o = optimizers[args.optimizer]
        return cls(
            case, n_train=args.n_train,
            n_skip=args.n_skip, lr=args.lr, plot=args.plot,
            out_dir=args.directory, opt_class=o
        )

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--n-train', '-t', dest='n_train',
            default=kw.get('n_train', 2500), type=int,
            help='Number of training iterations.'
        )
        p.add_argument(
            '--n-skip', dest='n_skip',
            default=kw.get('n_skip', 100), type=int,
            help='Number of iterations after which we print/update plots.'
        )
        p.add_argument(
            '--plot', dest='plot', action='store_true', default=False,
            help='Show a live plot of the results.'
        )
        p.add_argument(
            '--lr', dest='lr', default=kw.get('lr', 1e-2), type=float,
            help='Learning rate.'
        )
        p.add_argument(
            '--optimizer', dest='optimizer',
            default=kw.get('optimizer', 'Adam'),
            choices=['Adam', 'LBFGS'], help='Optimizer to use.'
        )
        p.add_argument(
            '-d', '--directory', dest='directory',
            default=kw.get('directory', None),
            help='Output directory (output files are dumped here).'
        )

    def __init__(self, case, n_train, n_skip=100, lr=1e-2, plot=True,
                 out_dir=None, opt_class=optim.Adam):
        '''Initializer

        Parameters
        -----------

        case: Case: The problem case being solved.
        n_train: int: Training steps
        n_skip: int: Print loss every so often.
        lr: float: Learming rate
        plot: bool: Plot live solution.
        out_dir: str: Output directory.
        '''
        self.case = case
        self.opt_class = opt_class
        self.errors = []
        self.loss = []
        self.time_taken = 0.0
        self.n_train = n_train
        self.n_skip = n_skip
        self.lr = lr
        self.plot = plot
        self.out_dir = out_dir

    def closure(self):
        opt = self.opt
        opt.zero_grad()
        loss = self.case.loss()
        loss.backward()
        self.loss.append(loss.item())
        return loss

    def solve(self):
        case = self.case
        n_train = self.n_train
        n_skip = self.n_skip
        opt = self.opt_class(case.nn.parameters(), lr=self.lr)
        self.opt = opt
        if self.plot:
            case.plot()

        start = time.perf_counter()
        for i in range(1, n_train+1):
            opt.step(self.closure)
            if i % n_skip == 0 or i == n_train:
                loss = self.case.loss()
                err = 0.0
                if self.plot:
                    err = case.plot()
                else:
                    err = case.get_error()
                self.errors.append(err)
                print(
                    f"Iteration ({i}/{n_train}): Loss={loss:.3e}, " +
                    f"error={err:.3e}"
                )
        time_taken = time.perf_counter() - start
        self.time_taken = time_taken
        print(f"Done. Took {time_taken:.3f} seconds.")
        if self.plot:
            plt.show()

    def save(self):
        dirname = self.out_dir
        if self.out_dir is None:
            print("No output directory set.  Skipping.")
            return
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print("Saving output to", dirname)
        fname = os.path.join(dirname, 'solver.npz')
        np.savez(
            fname, loss=self.loss, error=self.errors,
            time_taken=self.time_taken
        )
        self.case.save(dirname)


class DiffEq:
    def ode(self, x, u, ux, uxx):
        pass

    def exact(self, *args):
        pass


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
        self.k = torch.Tensor([1.0 + 2.0*np.log(2.0)])
        self.fac = self._sp(torch.Tensor([1.0]))

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

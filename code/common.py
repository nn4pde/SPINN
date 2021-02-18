import os
import time

import numpy as np
import torch
import torch.optim as optim

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


_device = torch.device("cpu")


def device(dev=None):
    '''Set/Get the device.
    '''
    global _device
    if dev is None:
        return _device
    else:
        _device = torch.device(dev)


def tensor(x, **kw):
    '''Returns a suitable device specific tensor.

    '''
    return torch.tensor(x, dtype=torch.float32, device=device(), **kw)


class PDE:
    @classmethod
    def from_args(cls, args):
        pass

    @classmethod
    def setup_argparse(cls, parser, **kw):
        pass

    def nodes(self):
        pass

    def fixed_nodes(self):
        pass

    def interior(self):
        pass

    def boundary(self):
        pass

    def plot_points(self):
        pass

    def eval_bc(self, problem):
        pass

    def pde(self, *args):
        pass

    def has_exact(self):
        return True

    def exact(self, *args):
        pass


class Problem:
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

    def loss(self):
        '''Return the loss.
        '''
        raise NotImplementedError()

    def get_plot_data(self):
        pass

    def get_error(self, **kw):
        pass

    def plot_solution(self):
        pass

    def plot(self):
        pass

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        pass

    def save(self, dirname):
        '''Save the model and results.

        '''
        pass

    def show(self):
        pass


class Optimizer:
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

    def __init__(self, problem, n_train, n_skip=100, lr=1e-2, plot=True,
                 out_dir=None, opt_class=optim.Adam):
        '''Initializer

        Parameters
        -----------

        problem: Problem: The problem case being solved.
        n_train: int: Training steps
        n_skip: int: Print loss every so often.
        lr: float: Learming rate
        plot: bool: Plot live solution.
        out_dir: str: Output directory.
        opt_class: Optimizer to use.
        '''
        self.problem = problem
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
        loss = self.problem.loss()
        loss.backward(retain_graph=True)
        self.loss.append(loss.item())
        return loss

    def solve(self):
        problem = self.problem
        n_train = self.n_train
        n_skip = self.n_skip
        opt = self.opt_class(problem.nn.parameters(), lr=self.lr)
        self.opt = opt
        if self.plot:
            problem.plot()

        start = time.perf_counter()
        for i in range(1, n_train+1):
            opt.step(self.closure)
            if i % n_skip == 0 or i == n_train:
                loss = self.problem.loss()
                err = 0.0
                if self.plot:
                    err = problem.plot()
                else:
                    err = problem.get_error()
                self.errors.append(err)
                if problem.pde.has_exact():
                    e_str = f", error={err:.3e}"
                else:
                    e_str = ''
                print(f"Iteration ({i}/{n_train}): Loss={loss:.3e}" + e_str)
        time_taken = time.perf_counter() - start
        self.time_taken = time_taken
        print(f"Done. Took {time_taken:.3f} seconds.")
        if self.plot:
            problem.show()

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
        self.problem.save(dirname)


class App:
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

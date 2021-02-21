# Base class for hybrid finite difference SPINN models

import os
import numpy as np
import matplotlib.pyplot as plt 
import torch

from common import device
from spinn1d import App1D, Plotter1D
from ode_base import BasicODE


class FDPlotter1D(Plotter1D):
    def show(self):
        pass

    def plot_solution(self):
        xn, pn = self.get_plot_data()
        pde = self.pde
        if self.plt1 is None:
            lines, = plt.plot(xn, pn, '-', label='computed')
            self.plt1 = lines
            if self.show_exact and pde.has_exact():
                yn = self.pde.exact(xn)
                lexact, = plt.plot(xn, yn, label='exact')
                self.lexact = lexact
            else:
                yn = pn
            plt.grid()
            plt.xlim(-0.1, 1.1)
            ymax, ymin = np.max(yn), np.min(yn)
            delta = (ymax - ymin)*0.5
            plt.ylim(ymin - delta, ymax + delta)
        else:
            self.plt1.set_data(xn, pn)
            if self.show_exact and pde.has_exact():
                yn = self.pde.exact(xn)
                self.lexact.set_data(xn, yn)
        return self.get_error(xn, pn)


class FDSPINN1D(BasicODE):
    def initial(self, x):
        return torch.zeros_like(x).to(device())

    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.samples, args.sample_frac,
                   args.dt, args.T, args.t_skip)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        super().setup_argparse(parser, **kw)

        p = parser
        p.add_argument(
            '--dt', dest='dt',
            default=kw.get('dt', 1e-3), type=float,
            help='Time step for finite difference scheme.'
        )
        p.add_argument(
            '--duration', '-T', dest='T',
            default=kw.get('T', 1.0), type=float,
            help='Duration of simulation'
        )
        p.add_argument(
            '--t-skip', dest='t_skip',
            default=kw.get('t_skip', 10), type=int,
            help='Iterations'
        )

    def __init__(self, n, ns, sample_frac=1.0, 
                 dt=1e-3, T=1.0, t_skip=10):
        super().__init__(n, ns, sample_frac)

        self.dt     = dt
        self.t      = self.dt
        self.T      = T
        self.t_skip = t_skip
        self.u0     = self.initial(self.xs)

    def pde(self, x, u, ux, uxx, u0, dt):
        pass

    def interior_loss(self, nn):
        xs = self.interior()
        u = nn(xs)
        u, ux, uxx = self._compute_derivatives(u, xs)
        res = self.pde(xs, u, ux, uxx, self.u0, self.dt)
        return (res**2).mean()


class AppFD1D(App1D):
    def run(self, args=None, **kw):
        parser = self.setup_argparse(**kw)
        args = parser.parse_args(args)

        if args.gpu:
            device("cuda")
        else:
            device("cpu")

        activation = self._get_activation(args)

        pde = self.pde_cls.from_args(args)
        self.pde = pde
        dev = device()
        nn = self.nn_cls.from_args(pde, activation, args).to(dev)
        self.nn = nn
        plotter = self.plotter_cls.from_args(pde, nn, args)
        self.plotter = plotter

        solver = self.optimizer.from_args(pde, nn, plotter, args)
        self.solver = solver

        n_itr = int(self.pde.T/self.pde.dt)

        for itr in range(n_itr):
            self.solver.solve()
            if itr > 0:
                self.pde.t += self.pde.dt

            with torch.no_grad():
                self.pde.u0 = self.nn(self.pde.xs)

            # if itr % self.pde.t_skip == 0:
                # FIX!

        dirname = self.solver.out_dir
        if self.solver.out_dir is None:
            print("No output directory set.  Skipping.")
        else:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            print("Saving output to", dirname)
            self.plotter.save(dirname)

        plt.show()

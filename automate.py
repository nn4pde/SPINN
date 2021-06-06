#!/usr/bin/env python
import os
from pathlib import Path

from automan.api import Problem, PySPHProblem
from automan.api import Automator, Simulation
import numpy as np
import torch
import matplotlib
from mayavi import mlab
matplotlib.use('pdf')
import matplotlib.pyplot as plt
mlab.options.offscreen = True

USE_GPU = False


fontsize = 32
font = {'size': fontsize}
matplotlib.rc('font', **font)


def figure(fsize=24):
    plt.figure(figsize=(12, 12))
    font = {'size': fsize}
    matplotlib.rc('font', **font)


def _plot_1d(problem, left_bdy=True, right_bdy=True):
    problem.make_output_dir()
    for case in problem.cases:
        res = np.load(case.input_path('results.npz'))
        nn_state = torch.load(case.input_path('model.pt'))

        plt.figure(figsize=(12, 12))
        plt.plot(
            res['x'], res['y_exact'],
            color='black', linewidth=6,
            label='Exact'
        )
        plt.plot(
            res['x'], res['y'], 'ro-',
            markersize=8, linewidth=3,
            label='SPINN'
        )
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')
        plt.legend()
        plt.grid()

        cen = nn_state['layer1.center'].tolist()
        if left_bdy:
            cen = [0.0] + cen
        if right_bdy:
            cen = cen + [1.0]

        plt.plot(
            cen, np.zeros_like(cen),
            'bo', markersize=8, label='Nodes'
        )

        plt.tight_layout()
        plt.savefig(problem.output_path(
            f"{problem.get_name()}_n_{case.params['nodes']}.pdf"
        ))
        plt.close()


def _mlab_figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(1000, 1000)):
    f = mlab.figure(fgcolor=fgcolor, bgcolor=bgcolor, size=size)
    return f


def _plot_mlab_nodes(state):
    x = state['layer1.x'].numpy()
    y = state['layer1.y'].numpy()
    w = state['layer1.h'].numpy()
    p = mlab.points3d(
        x, y, np.zeros_like(x), color=(1, 0, 0), scale_factor=0.05
    )
    p1 = mlab.points3d(
        x, y, np.zeros_like(x), w[:len(x)], color=(1.0, 0.0, 0.0),
        mode='2dcircle', scale_factor=1.0, opacity=0.8
    )
    p1.glyph.glyph_source.glyph_source.resolution = 40
    return p, p1


def plot_solution_nodes(plot_data, state, fname):
    sd = plot_data
    f = _mlab_figure()
    s = mlab.surf(sd['x'], sd['y'], sd['u'], warp_scale=0.0,
                  colormap='viridis', opacity=0.4)
    p, p1 = _plot_mlab_nodes(state)
    mlab.axes(s, xlabel='x', ylabel='t', y_axis_visibility=False)
    s.scene.z_plus_view()
    mlab.savefig(fname)
    mlab.close()


def plot_3d_solution(plot_data, warp_scale=0.5):
    sd = plot_data
    f = _mlab_figure()
    s = mlab.surf(sd['x'], sd['y'], sd['u'], warp_scale=warp_scale,
                  colormap='viridis')
    ex = mlab.surf(sd['x'], sd['y'], sd['u_exact'],
                   warp_scale=warp_scale, representation='wireframe',
                   color=(0, 0, 0))
    mlab.outline(s)
    mlab.axes(s, xlabel='x', ylabel='t', zlabel='u')
    s.scene.isometric_view()


def plot_ug_solution(vtu, u):
    src = mlab.pipeline.open(vtu)
    ug = src.reader.output
    ug.point_data.scalars = u
    s = mlab.pipeline.surface(src, colormap='viridis')
    s.scene.z_plus_view()
    return s


def softplus(x):
    return np.log(1.0 + np.exp(x))


def relu(x):
    return np.maximum(0.0, x)


def hat_softplus_1d(x):
    return softplus(
        1.0 + 2.0*np.log(2.0) - softplus(x) - softplus(-x)
    ) / softplus(1.0)


def gaussian_1d(x):
    return np.exp(-0.5*x*x)


def hat_softplus_2d(x, y):
    return softplus(
        1.0 + 4.0*np.log(2.0) - softplus(x) - softplus(-x)
        - softplus(y) - softplus(-y)
    )/softplus(1.0)


class Misc(Problem):
    def get_name(self):
        return 'misc'

    def setup(self):
        self.cases = []

    def _plot_softplus_relu(self):
        fname = self.output_path('SoftPlus_ReLU.pdf')
        xs = np.linspace(-8.0, 5.0, 200)
        figure(fsize=32)
        plt.plot(xs, relu(xs), 'k-', linewidth=8, label='ReLU')
        plt.plot(xs, softplus(xs), 'b-', linewidth=6, label='SoftPlus')
        plt.xlabel(r'$x$', fontsize=32)
        plt.ylabel(r'$y$', fontsize=32)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_softplus_hat(self):
        xs = np.linspace(-12.0, 12.0, 200)
        figure(fsize=32)
        plt.plot(xs, gaussian_1d(xs), 'k--', linewidth=3, label='Gaussian')
        plt.plot(xs, hat_softplus_1d(xs), 'b-',
                 linewidth=6, label='Softplus hat')
        plt.xlabel(r'$x$', fontsize=32)
        plt.ylabel(r'$y$', fontsize=32)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        fname = self.output_path('SoftPlus_Hat_1d.pdf')
        plt.savefig(fname)
        plt.close()

    def _plot_softplus_hat_2d(self):
        xs, ys = np.mgrid[-10.0:10.0:0.1, -10.0:10.0:0.1]
        zs = hat_softplus_2d(xs, ys)
        f = _mlab_figure(size=(1200, 900))
        mlab.surf(xs, ys, zs, warp_scale='auto')
        mlab.axes(ranges=[-10.0, 10.0, -10.0, 10.0, 0.0, 1.0])
        mlab.view(160, 75, distance=350, focalpoint=[0.0, 100.0, 0.0])
        fname = self.output_path('SoftPlus_Hat_2d.png')
        mlab.savefig(fname)

    def run(self):
        self.make_output_dir()
        self._plot_softplus_relu()
        self._plot_softplus_hat()
        self._plot_softplus_hat_2d()


class ODE1(Problem):
    def get_name(self):
        return 'ode1'

    def setup(self):
        base_cmd = (
            'python3 code/ode1.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{i}'),
                base_command=base_cmd,
                nodes=i, samples=6*i,
                n_train=100000,
                lr=1e-4,
                tol=1e-6
            )
            for i in (1, 3, 7)
        ]

    def run(self):
        _plot_1d(self)


class ODE2(Problem):
    def get_name(self):
        return 'ode2'

    def setup(self):
        base_cmd = (
            'python3 code/ode2.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{i}'),
                base_command=base_cmd,
                nodes=i, samples=15*i,
                sample_frac=0.2,
                n_train=100000,
                lr=1e-3,
                tol=1e-3
            )
            for i in (3, 5, 7)
        ]

    def run(self):
        _plot_1d(self, right_bdy=False)


class ODE3(Problem):
    def get_name(self):
        return 'ode3'

    def setup(self):
        base_cmd = (
            'python3 code/ode3.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{i}'),
                base_command=base_cmd,
                nodes=i, samples=20*i,
                n_train=100000,
                lr=1e-4,
                tol=1e-6
            )
            for i in (1, 3, 7)
        ]

    def run(self):
        _plot_1d(self)


def _plot_ode_conv(problem, n_nodes, pname='ode',
                   left_bdy=True, right_bdy=True):
    problem.make_output_dir()

    L1s = []
    L2s = []
    Linfs = []

    acts = []
    for case in problem.cases:
        acts.append(case.params['activation'])

    for case in problem.cases:
        ## Plot SPINN vs exact solution
        res = np.load(case.input_path('results.npz'))
        nn_state = torch.load(case.input_path('model.pt'))

        plt.figure(figsize=(12,12))
        plt.plot(
            res['x'], res['y_exact'],
            color='black', linewidth=6,
            label='Exact'
        )
        plt.plot(
            res['x'], res['y'], 'ro-',
            markersize=8, linewidth=3,
            label='SPINN'
        )
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')
        plt.legend()
        plt.grid()

        ## Plot nodal positions
        cen = nn_state['layer1.center'].tolist()
        if left_bdy:
            cen = [0.0] + cen
        if right_bdy:
            cen = cen + [1.0]

        plt.plot(
            cen, np.zeros_like(cen),
            'bo', markersize=8, label='Nodes'
        )

        plt.tight_layout()
        plt.savefig(problem.output_path(
            f"{pname}_{case.params['activation']}_n_{n_nodes}.pdf"
        ))
        plt.close()

        ## Save errors
        res = np.load(case.input_path('solver.npz'))
        L1s.append(res['error_L1'])
        L2s.append(res['error_L2'])
        Linfs.append(res['error_Linf'])

    ## Plot L1 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(acts)):
        plt.plot(L1s[i], '-', linewidth=4, label=f'{acts[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_1$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_L1_error_n_{n_nodes}.pdf'
    ))
    plt.close()

    ## Plot L2 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(acts)):
        plt.plot(L2s[i], '-', linewidth=4, label=f'{acts[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_2$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_L2_error_n_{n_nodes}.pdf'
    ))
    plt.close()

    ## Plot Linf error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(acts)):
        plt.plot(Linfs[i], '-', linewidth=4, label=f'{acts[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_{\infty}$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_Linf_error_n_{n_nodes}.pdf'
    ))
    plt.close()


class ODE1Conv1(Problem):
    def get_name(self):
        return 'ode1_conv_1'

    def setup(self):
        self.n = 1
        self.ns = 6*self.n

        base_cmd = (
            'python3 code/ode1.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n_{self.n}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                n_train=1000000, n_skip=100,
                lr=1e-4,
                tol=1e-6,
                activation=activation
            )
            for activation in ('kernel', 'softplus', 'gaussian')
        ]

    def run(self):
        _plot_ode_conv(self, self.n, 'ode1')


class ODE1Conv3(Problem):
    def get_name(self):
        return 'ode1_conv_3'

    def setup(self):
        self.n = 3
        self.ns = 6*self.n

        base_cmd = (
            'python3 code/ode1.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n_{self.n}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                n_train=1000000, n_skip=100,
                lr=1e-4,
                tol=1e-6,
                activation=activation
            )
            for activation in ('kernel', 'softplus', 'gaussian')
        ]

    def run(self):
        _plot_ode_conv(self, self.n, 'ode1')


class ODE1Conv7(Problem):
    def get_name(self):
        return 'ode1_conv_7'

    def setup(self):
        self.n = 7
        self.ns = 6*self.n

        base_cmd = (
            'python3 code/ode1.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n_{self.n}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                n_train=1000000, n_skip=100,
                lr=1e-4,
                tol=1e-6,
                activation=activation
            )
            for activation in ('kernel', 'softplus', 'gaussian')
        ]

    def run(self):
        _plot_ode_conv(self, self.n, 'ode1')


class ODE3Conv1(Problem):
    def get_name(self):
        return 'ode3_conv_1'

    def setup(self):
        self.n = 1
        self.ns = 20*self.n

        base_cmd = (
            'python3 code/ode3.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n_{self.n}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                n_train=100000, n_skip=1,
                lr=1e-4,
                tol=1e-6,
                activation=activation
            )
            for activation in ('kernel', 'softplus', 'gaussian')
        ]

    def run(self):
        _plot_ode_conv(self, self.n, 'ode3')


class ODE3Conv3(Problem):
    def get_name(self):
        return 'ode3_conv_3'

    def setup(self):
        self.n = 3
        self.ns = 20*self.n

        base_cmd = (
            'python3 code/ode3.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n_{self.n}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                n_train=100000, n_skip=1,
                lr=1e-4,
                tol=1e-6,
                activation=activation
            )
            for activation in ('kernel', 'softplus', 'gaussian')
        ]

    def run(self):
        _plot_ode_conv(self, self.n, 'ode3')


def _plot_ode_conv_sampling(problem, n_nodes, pname='ode',
                            left_bdy=True, right_bdy=True):
    problem.make_output_dir()

    L1s = []
    L2s = []
    Linfs = []

    fs = []
    for case in problem.cases:
        fs.append(case.params['sample_frac'])

    for case in problem.cases:
        ## Plot SPINN vs exact solution
        res = np.load(case.input_path('results.npz'))
        nn_state = torch.load(case.input_path('model.pt'))

        plt.figure(figsize=(12,12))
        plt.plot(
            res['x'], res['y_exact'],
            color='black', linewidth=6,
            label='Exact'
        )
        plt.plot(
            res['x'], res['y'], 'ro-',
            markersize=8, linewidth=3,
            label='SPINN'
        )
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')
        plt.legend()
        plt.grid()

        ## Plot nodal positions
        cen = nn_state['layer1.center'].tolist()
        if left_bdy:
            cen = [0.0] + cen
        if right_bdy:
            cen = cen + [1.0]

        plt.plot(
            cen, np.zeros_like(cen),
            'bo', markersize=8, label='Nodes'
        )

        plt.tight_layout()
        sample_frac = f"{case.params['sample_frac']}".replace('.', 'p')
        plt.savefig(problem.output_path(
            f"{pname}_n_{n_nodes}_f_{sample_frac}.pdf"
        ))
        plt.close()

        ## Save errors
        res = np.load(case.input_path('solver.npz'))
        L1s.append(res['error_L1'])
        L2s.append(res['error_L2'])
        Linfs.append(res['error_Linf'])

    ## Plot L1 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(fs)):
        plt.plot(L1s[i], '-', linewidth=4, label=f'f={fs[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_1$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_L1_error_n_{n_nodes}_f.pdf'
    ))
    plt.close()

    ## Plot L2 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(fs)):
        plt.plot(L2s[i], '-', linewidth=4, label=f'f={fs[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_2$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_L2_error_n_{n_nodes}_f.pdf'
    ))
    plt.close()

    ## Plot Linf error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(fs)):
        plt.plot(Linfs[i], '-', linewidth=4, label=f'f={fs[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_{\infty}$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_Linf_error_n_{n_nodes}_f.pdf'
    ))
    plt.close()


def _plot_ode_rep_sampling(problem, n_nodes, pname='ode',
                           left_bdy=True, right_bdy=True):
    problem.make_output_dir()

    Linfs = []

    plt.figure(figsize=(12,12))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u(x)$')
    plt.grid()

    count = 0
    sample_frac = 0

    for case in problem.cases:
        count += 1
        if count == 1:
            sample_frac = case.params['sample_frac']

        ## Plot SPINN vs exact solution
        res = np.load(case.input_path('results.npz'))
        nn_state = torch.load(case.input_path('model.pt'))

        if count == 1:
            plt.plot(
                res['x'], res['y_exact'], 'k-',
                linewidth=6,
                label='Exact'
            )

        plt.plot(
            res['x'], res['y'], 'o-',
            markersize=8, linewidth=3,
            label=f'Trial {count}'
        )

        ## Save errors
        res = np.load(case.input_path('solver.npz'))
        Linfs.append(res['error_Linf'])

    plt.legend()
    plt.tight_layout()
    sample_frac = f"{sample_frac}".replace('.', 'p')
    plt.savefig(problem.output_path(
        f"{pname}_n_{n_nodes}_f_{sample_frac}_rep.pdf"
    ))
    plt.close()

    ## Plot Linf error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(problem.cases)):
        plt.plot(Linfs[i], '-', linewidth=4, label=f'Trial {i+1}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_{\infty}$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_Linf_error_n_{n_nodes}_rep.pdf'
    ))
    plt.close()


class ODE2Conv5(Problem):
    def get_name(self):
        return 'ode2_conv_5'

    def setup(self):
        self.n = 5
        self.ns = 15*self.n

        base_cmd = (
            'python3 code/ode2.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{self.n}_f_{f}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                sample_frac=f,
                n_train=1000000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation='gaussian'
            )
            for f in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
        ]

    def run(self):
        _plot_ode_conv_sampling(self, self.n, 'ode2', right_bdy=False)


class ODE2Rep5(Problem):
    def get_name(self):
        return 'ode2_rep_5'

    def setup(self):
        self.n = 5
        self.ns = 15*self.n

        base_cmd = (
            'python3 code/ode2.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{self.n}_f_{f}_{i}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                sample_frac=f,
                n_train=1000000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation='gaussian'
            )
            for i, f in enumerate((0.2, 0.2, 0.2, 0.2, 0.2))
        ]

    def run(self):
        _plot_ode_rep_sampling(self, self.n, 'ode2', right_bdy=False)


def _plot_1d_err(problem, pname, n_nodes):
    problem.make_output_dir()
    for case in problem.cases:
        res = np.load(case.input_path('solver.npz'))
        Linfs = res['error_Linf']

        plt.figure(figsize=(12,12))
        plt.xscale('log')
        plt.yscale('log')

        plt.plot(Linfs, '-', linewidth=4)

        plt.xlabel('Iterations')
        plt.ylabel(r'$L_{\infty}$ error')
        plt.grid()
        plt.tight_layout()
        plt.savefig(problem.output_path(
            f'{pname}_Linf_error_n_{n_nodes}.pdf'
        ))
        plt.close()


class ODE1Var(Problem):
    def get_name(self):
        return 'ode1_var'

    def setup(self):
        base_cmd = (
            'python3 code/ode1_var.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{i}'),
                base_command=base_cmd,
                nodes=i, samples=500,
                n_train=100000,
                lr=1e-4,
                tol=-10.0
            )
            for i in (5,)
        ]

    def run(self):
        _plot_1d(self)
        _plot_1d_err(self, 'ode1_var', 5)


class ODE3Var(Problem):
    def get_name(self):
        return 'ode3_var'

    def setup(self):
        base_cmd = (
            'python3 code/ode3_var.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{i}'),
                base_command=base_cmd,
                nodes=i, samples=250,
                n_train=100000,
                lr=1e-4,
                tol=-10.0
            )
            for i in (5,)
        ]

    def run(self):
        _plot_1d(self)
        _plot_1d_err(self, 'ode3_var', 5)


def _plot_1d_fourier(problem, left_bdy=True, right_bdy=True):
    problem.make_output_dir()
    for case in problem.cases:
        res = np.load(case.input_path('results.npz'))
        nn_state = torch.load(case.input_path('model.pt'))

        plt.figure(figsize=(12,12))
        plt.plot(
            res['x'], res['y_exact'],
            color='black', linewidth=6,
            label='Exact'
        )
        plt.plot(
            res['x'], res['y'], 'ro-',
            markersize=8, linewidth=3,
            label='Fourier-SPINN'
        )
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(problem.output_path(
            f"{problem.get_name()}_m_{case.params['modes']}.pdf"
        ))
        plt.close()


class ODE1Fourier(Problem):
    def get_name(self):
        return 'ode1_fourier'

    def setup(self):
        base_cmd = (
            'python3 code/ode1_fourier.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{i}'),
                base_command=base_cmd,
                modes=i, samples=20,
                n_train=100000,
                lr=1e-4,
                tol=1e-6
            )
            for i in (10,)
        ]

    def run(self):
        _plot_1d_fourier(self)
        _plot_1d_err(self, 'ode1_fourier', 10)


class ODE3Fourier(Problem):
    def get_name(self):
        return 'ode3_fourier'

    def setup(self):
        base_cmd = (
            'python3 code/ode3_fourier.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{i}'),
                base_command=base_cmd,
                modes=i, samples=50,
                n_train=100000,
                lr=1e-4,
                tol=1e-6
            )
            for i in (50,)
        ]

    def run(self):
        _plot_1d_fourier(self)
        _plot_1d_err(self, 'ode3_fourier', 50)


class ODE3Comp(Problem):
    def get_name(self):
        return 'ode3_comp'

    def setup(self):
        base_cmd_1 = (
            'python3 code/ode3.py -d $output_dir '
        )
        base_cmd_2 = (
            'python3 code/ode3_var.py -d $output_dir '
        )
        base_cmd_3 = (
            'python3 code/ode3_fourier.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path('strong_n_3'),
                base_command=base_cmd_1,
                nodes=3, samples=60,
                n_train=20000,
                lr=1e-3,
                tol=1e-3
            ),
            Simulation(
                root=self.input_path('weak_n_5'),
                base_command=base_cmd_2,
                nodes=5, samples=200,
                n_train=5000,
                lr=5e-3,
                tol=-10.0
            ),
            Simulation(
                root=self.input_path('fourier_m_50'),
                base_command=base_cmd_3,
                modes=50, samples=50,
                n_train=5000,
                lr=1e-2,
                tol=1e-4
            )
        ]

    def run(self):
        self.make_output_dir()

        left_bdy = True
        right_bdy = True

        ## Hack - hard coded order in which cases are run
        case_count = 0
        case_label=['SPINN', 'Var-SPINN', 'Fourier-SPINN']
        case_color=['red', 'blue', 'green']
        case_style=['o-', 's', '^']
        case_every=[1, 3, 2]

        plt.figure(figsize=(12,12))
        plt.xlim(-0.1, 1.3)

        for case in self.cases:
            res = np.load(case.input_path('results.npz'))
            nn_state = torch.load(case.input_path('model.pt'))

            if case_count == 0:
                plt.plot(
                    res['x'], res['y_exact'],
                    color='black', linewidth=6,
                    label='Exact'
                )

            plt.plot(
                res['x'], res['y'],
                case_style[case_count],
                color=case_color[case_count],
                markersize=10, linewidth=3,
                markevery=case_every[case_count],
                label=case_label[case_count]
            )
            plt.xlabel(r'$x$')
            plt.ylabel(r'$u(x)$')
            plt.legend()
            plt.grid()

            if case_count == 0:
                cen = nn_state['layer1.center'].tolist()
                if left_bdy:
                    cen = [0.0] + cen
                if right_bdy:
                    cen = cen + [1.0]

                plt.plot(
                    cen, np.zeros_like(cen),
                    'yP', markersize=10, label='SPINN nodes'
                )
            elif case_count == 1:
                cen = nn_state['layer1.center'].tolist()
                if left_bdy:
                    cen = [0.0] + cen
                if right_bdy:
                    cen = cen + [1.0]

                plt.plot(
                    cen, np.zeros_like(cen),
                    'cX', markersize=10, label='Var-SPINN nodes'
                )

            case_count += 1

        plt.tight_layout()
        plt.savefig(self.output_path(
            f"{self.get_name()}.pdf"
        ))
        plt.close()


def _plot_pde_conv(problem, n_nodes):
    problem.make_output_dir()

    L1s = []
    L2s = []
    Linfs = []

    acts = []
    for case in problem.cases:
        acts.append(case.params['activation'])

    for case in problem.cases:
        ## Plot SPINN vs exact solution
        res = np.load(case.input_path('results.npz'))
        x = res['x']
        y = res['y']
        u = res['u']
        u_exact = res['u_exact']

        mlab.figure(size=(700, 700), bgcolor=(1,1,1), fgcolor=(0,0,0))
        m_plt = mlab.surf(x, y, u, colormap='viridis', opacity=1.0)
        mlab.surf(x, y, u_exact, colormap='viridis',
                  representation='wireframe')
        mlab.colorbar(m_plt)
        mlab.savefig(problem.output_path(f"sine2d_{case.params['activation']}_n_{n_nodes}.pdf"))
        mlab.close()

        ## Save errors
        res = np.load(case.input_path('solver.npz'))
        L1s.append(res['error_L1'])
        L2s.append(res['error_L2'])
        Linfs.append(res['error_Linf'])

    ## Plot L1 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(acts)):
        plt.plot(L1s[i], '-', linewidth=4, label=f'{acts[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_1$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'sine2d_L1_error_n_{n_nodes}.pdf'))
    plt.close()

    ## Plot L2 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(acts)):
        plt.plot(L2s[i], '-', linewidth=4, label=f'{acts[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_2$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'sine2d_L2_error_n_{n_nodes}.pdf'))
    plt.close()

    ## Plot Linf error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(acts)):
        plt.plot(Linfs[i], '-', linewidth=4, label=f'{acts[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_{\infty}$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'sine2d_Linf_error_n_{n_nodes}.pdf'))
    plt.close()


class Poisson2DSineConv(Problem):
    def get_name(self):
        return 'poisson2d_sine_conv'

    def setup(self):
        self.n = 100
        self.ns = 400

        base_cmd = (
            'python3 code/poisson2d_sine.py -d $output_dir'
        )
        kwargs = dict(gpu=None) if USE_GPU else {}

        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n_{self.n}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                n_train=25000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation=activation,
                **kwargs
            )
            for activation in ('gaussian', 'softplus', 'kernel')
        ]

    def run(self):
        _plot_pde_conv(self, self.n)


def _plot_pde_conv_nodes(problem):
    problem.make_output_dir()

    L1s = []
    L2s = []
    Linfs = []

    n_nodes = []
    for case in problem.cases:
        n_nodes.append(case.params['nodes'])

    for case in problem.cases:
        ## Plot SPINN vs exact solution
        res = np.load(case.input_path('results.npz'))
        x = res['x']
        y = res['y']
        u = res['u']
        u_exact = res['u_exact']

        mlab.figure(size=(700, 700), bgcolor=(1,1,1), fgcolor=(0,0,0))
        m_plt = mlab.surf(x, y, u, colormap='viridis', opacity=1.0)
        mlab.surf(x, y, u_exact, colormap='viridis',
                  representation='wireframe')
        mlab.colorbar(m_plt)
        mlab.savefig(problem.output_path(f"sine2d_n_{case.params['nodes']}.pdf"))
        mlab.close()

        ## Save errors
        res = np.load(case.input_path('solver.npz'))
        L1s.append(res['error_L1'])
        L2s.append(res['error_L2'])
        Linfs.append(res['error_Linf'])

    ## Plot L1 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(n_nodes)):
        plt.plot(L1s[i], '-', linewidth=4, label=f'n~{n_nodes[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_1$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'sine2d_L1_error.pdf'))
    plt.close()

    ## Plot L2 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(n_nodes)):
        plt.plot(L2s[i], '-', linewidth=4, label=f'n~{n_nodes[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_2$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'sine2d_L2_error.pdf'))
    plt.close()

    ## Plot Linf error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(n_nodes)):
        plt.plot(Linfs[i], '-', linewidth=4, label=f'n~{n_nodes[i]}')

    plt.xlabel('Iterations')
    plt.ylabel(r'$L_{\infty}$ error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'sine2d_Linf_error.pdf'))
    plt.close()


class Poisson2DSineNodes(Problem):
    def get_name(self):
        return 'poisson2d_sine_nodes'

    def setup(self):
        base_cmd = (
            'python3 code/poisson2d_sine.py -d $output_dir'
        )
        kwargs = dict(gpu=None) if USE_GPU else {}

        self.cases = [
            Simulation(
                root=self.input_path(f'n_{n}'),
                base_command=base_cmd,
                nodes=n, samples=4*n,
                n_train=20000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation='softplus',
                **kwargs
            )
            for n in (25, 50, 75, 100)
        ]

    def run(self):
        _plot_pde_conv_nodes(self)


def _plot_pde_conv_nodes_fem(problem):
    problem.make_output_dir()

    L1s = []
    L2s = []
    Linfs = []

    n_nodes = []
    for case in problem.cases:
        n_nodes.append(case.params['nodes'])

    for case in problem.cases:
        ## Collect errors
        res = np.load(case.input_path('results.npz'))
        L1s.append(res['L1'].item())
        L2s.append(res['L2'].item())
        Linfs.append(res['Linf'].item())

    ## Plot L1 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(n_nodes, L1s, 'b-',linewidth=6)
    plt.xlabel('Number of nodes')
    plt.ylabel(r'$L_1$ error')
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'square_slit_L1_error.pdf'))
    plt.close()

    ## Plot L2 error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(n_nodes, L2s, 'b-',linewidth=6)
    plt.xlabel('Number of nodes')
    plt.ylabel(r'$L_2$ error')
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'square_slit_L2_error.pdf'))
    plt.close()

    ## Plot Linf error as function of iteration
    plt.figure(figsize=(12,12))
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(n_nodes, Linfs, 'b-',linewidth=6)
    plt.xlabel('Number of nodes')
    plt.ylabel(r'$L_{\infty}$ error')
    plt.grid()
    plt.tight_layout()
    plt.savefig(problem.output_path(f'square_slit_Linf_error.pdf'))
    plt.close()


class SquareSlit(Problem):
    def get_name(self):
        return 'square_slit'

    def setup(self):
        base_cmd = (
            'python3 code/poisson2d_square_slit.py -d $output_dir'
        )
        kwargs = dict(gpu=None) if USE_GPU else {}
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{n}'),
                base_command=base_cmd,
                nodes=n, samples=3*n,
                n_train=10000, n_skip=100,
                lr=1e-3,
                tol=1e-4,
                activation='softplus',
                **kwargs
            )
            for n in (25, 50, 100, 200, 500)
        ]

    def run(self):
        self.make_output_dir()
        _plot_pde_conv_nodes_fem(self)
        self._plot_solution_nodes()

    def _plot_solution_nodes(self):
        case = self.cases[3]
        sd = np.load(case.input_path('results.npz'))
        state = torch.load(case.input_path('model.pt'))
        vtu_fname = os.path.join('code', 'fem', 'poisson_solution000000.vtu')
        fname = self.output_path('solution.png')
        f = _mlab_figure()
        s = plot_ug_solution(vtu_fname, sd['u'])
        cb = mlab.colorbar(s, title='u')
        mlab.axes(s, xlabel='x', ylabel='y', y_axis_visibility=False)
        mlab.move(up=-0.25)
        mlab.savefig(fname)
        fname = self.output_path('sol_centers.png')
        s.actor.property.opacity = 0.4
        p, p1 = _plot_mlab_nodes(state)
        cb.visible = False
        s.scene.z_plus_view()
        mlab.move(forward=0.5)
        mlab.savefig(fname)
        mlab.close()


class Irregular(Problem):
    def get_name(self):
        return 'irregular'

    def setup(self):
        base_cmd = (
            'python3 code/poisson2d_irreg_dom.py -d $output_dir'
        )
        kwargs = dict(gpu=None) if USE_GPU else {}
        self.cases = [
            Simulation(
                root=self.input_path(f'irregular_domain'),
                base_command=base_cmd,
                lr=1e-3,
                **kwargs
            )
        ]

    def _plot_fem_solution(self):
        vtu_fname = os.path.join('code', 'fem', 'poisson_irregular000000.vtu')
        fname = self.output_path('fem_solution.png')
        f = _mlab_figure()
        src = mlab.pipeline.open(vtu_fname)
        s = mlab.pipeline.surface(src, colormap='viridis')
        cb = mlab.colorbar(s, title='u')
        s.scene.z_plus_view()
        mlab.savefig(fname)
        mlab.close()

    def run(self):
        self.make_output_dir()
        self._plot_fem_solution()
        case = self.cases[0]
        sd = np.load(case.input_path('results.npz'))
        state = torch.load(case.input_path('model.pt'))
        f = _mlab_figure()
        x, y = sd['x'], sd['y']
        pts = mlab.points3d(
            x, y, -0.01*np.ones_like(y), sd['u'],
            colormap='viridis', mode='2dcircle', scale_mode='none',
            scale_factor=0.0125
        )
        pts.glyph.glyph_source.glyph_source.filled = True
        pts.glyph.glyph_source.glyph_source.resolution = 20
        pts.actor.property.point_size = 10
        cb = mlab.colorbar(pts, title='u')
        pts.scene.z_plus_view()
        fname = self.output_path('spinn_solution.png')
        mlab.savefig(fname)
        fname = self.output_path('sol_centers.png')
        pts.actor.property.opacity = 0.4
        cb.visible = False
        p, p1 = _plot_mlab_nodes(state)
        p.glyph.glyph.scale_factor = 0.01
        mlab.move(forward=2.5)
        mlab.savefig(fname)
        mlab.close()


# Irregular domain
# python poisson2d_irreg_dom.py --plot --lr 1e-3 --gpu


def plot_centers(x, y, w):
    figure()
    a = plt.gca()
    circles = [
        plt.Circle((xi, yi), radius=wi, linewidth=1, fill=False,
                   color='blue', alpha=0.2)
        for xi, yi, wi in zip(x, y, w[:len(x)])
    ]
    c = matplotlib.collections.PatchCollection(
        circles, match_original=True
    )
    a.add_collection(c)
    plt.scatter(x, y, s=100, marker='o')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')


class Cavity(Problem):
    def get_name(self):
        return 'cavity'

    def _n_nodes(self, nn_state):
        return len(nn_state['layer1.x'])

    def setup(self):
        base_cmd = (
            'python3 code/cavity.py -d $output_dir '
        )

        self.cases = [
            Simulation(
                root=self.input_path(f'n{i}'),
                base_command=base_cmd,
                nodes=i, samples=5000,
                n_train=10000,
                lr=5e-4,
                tol=1e-3,
                sample_frac=0.1,
            )
            for i in (100, 200, 300)
        ]

    def get_ghia_data(self):
        fname = os.path.join('code', 'data', 'ghia_re_100.txt')
        x, v, y, u = np.loadtxt(fname, unpack=True)
        return x, v, y, u

    def plot_comparison(self):
        xg, vg, yg, ug = self.get_ghia_data()

        matplotlib.rc('font', **font)
        f1, ax1 = plt.subplots(figsize=(12, 12))
        f2, ax2 = plt.subplots(figsize=(12, 12))
        ax1.plot(ug, yg, 'o', color='red', label='Ghia et al.')
        ax1.set_xlabel(r'$u$')
        ax1.set_ylabel(r'$y$')

        ax2.plot(xg, vg, 'o', color='red', label='Ghia et al.')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$v$')
        ls = [':', '--', '-']
        for i, case in enumerate(self.cases):
            nn_state = torch.load(case.input_path('model.pt'))
            nodes = self._n_nodes(nn_state)
            res = np.load(case.input_path('results.npz'))
            xc, uc, vc = res['xc'], res['uc'], res['vc']
            ax1.plot(uc, xc, ls[i], lw=3,
                     label='SPINN (%d nodes)' % nodes)
            ax2.plot(xc, vc, ls[i], lw=3,
                     label='SPINN (%d nodes)' % nodes)

        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        f1.savefig(self.output_path('u_vs_y.pdf'))
        f2.savefig(self.output_path('x_vs_v.pdf'))
        plt.close(f1)
        plt.close(f2)

    def plot_result(self):
        for case in self.cases:
            nn_state = torch.load(case.input_path('model.pt'))
            nodes = self._n_nodes(nn_state)
            res = np.load(case.input_path('results.npz'))
            x, y, U = res['x'], res['y'], res['u']
            u, v = U[..., 0], U[..., 1]
            vmag = np.sqrt(u*u + v*v)
            figure()
            plt.contourf(x, y, vmag, levels=25)
            plt.colorbar()
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.axis('equal')
            fname = self.output_path('vmag_n%d.pdf' % (nodes))
            plt.savefig(fname)
            plt.close()

    def run(self):
        self.make_output_dir()
        self.plot_result()
        self.plot_comparison()


class CavityPySPH(PySPHProblem):
    def get_name(self):
        return 'cavity_pysph'

    def setup(self):
        cmd = 'pysph run cavity --re 100 --scheme edac --tf 15'
        self.cases = [
            Simulation(
                root=self.input_path(f'nx_{i}'),
                base_command=cmd, nx=i
            )
            for i in (10, 25, 50)
        ]

    def run(self):
        self.make_output_dir()
        fname = os.path.join('code', 'data', 'ghia_re_100.txt')
        xg, vg, yg, ug = np.loadtxt(fname, unpack=True)
        matplotlib.rc('font', **font)
        f1, ax1 = plt.subplots(figsize=(12, 12))
        f2, ax2 = plt.subplots(figsize=(12, 12))

        ax1.plot(ug, yg, 'o', color='red', label='Ghia et al.')
        ax1.set_xlabel(r'$u$')
        ax1.set_ylabel(r'$y$')
        ax2.plot(xg, vg, 'o', color='red', label='Ghia et al.')
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$v$')
        ls = [':', '--', '-']
        for i, case in enumerate(self.cases):
            nx = case.params['nx']
            ax1.plot(case.data['u_c'], case.data['x'],
                     ls[i], lw=3, label=f'SPH ({nx}x{nx})')
            ax2.plot(case.data['x'], case.data['v_c'],
                     ls[i], lw=3, label=f'SPH ({nx}x{nx})')
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        f1.savefig(self.output_path('u_vs_y.pdf'))
        f2.savefig(self.output_path('x_vs_v.pdf'))
        plt.close(f1)
        plt.close(f2)


def get_results(case, times):
    files = sorted(Path(case.input_path()).glob('results_*.npz'))
    mfiles = sorted(Path(case.input_path()).glob('model_*.pt'))
    data = [np.load(str(f)) for f in files]
    result = []
    models = []
    count = 0
    for time in times:
        while count < len(data):
            d = data[count]
            count += 1
            if abs(d['t'] - time) < 1e-10:
                result.append(d)
                models.append(torch.load(mfiles[count-1]))
                break

    assert len(result) == len(times), "Insufficient output data?"
    return result, models


def plot_fd_centers(problem, models, times):
    figure()
    for t, model in zip(times, models):
        x = model['layer1.center'].numpy()
        plt.scatter(x, t*np.ones_like(x), s=100, color='blue', alpha=0.7)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.grid()
    fname = problem.output_path('centers_fd.pdf')
    plt.savefig(fname)


class BurgersFD(Problem):
    def get_name(self):
        return 'burgers_fd'

    def _n_nodes(self, nn_state):
        return len(nn_state['layer1.center'])

    def setup(self):
        base_cmd = (
            'python3 code/burgers1d_fd_spinn.py -d $output_dir '
        )

        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n{i}'),
                base_command=base_cmd,
                nodes=i, samples=400,
                n_train=5000,
                lr=1e-4,
                tol=1e-6,
                dt=0.01,
                activation=activation,
                sample_frac=1.0,
            )
            for i in (20, 40, 80) for activation in ('gaussian',)
        ]

    def _plot_solution(self, case, nodes, exact):
        data, models = get_results(case, [0.1, 0.3, 0.6, 1.0])
        colors = ['violet', 'blue', 'green', 'red']
        x_ex = exact['x']
        u_ex = exact['u']
        figure()

        for count, i in enumerate((1, 3, 6, 10)):
            t = i*0.1
            res = data[count]
            plt.plot(
                res['x'], res['y'], 'o-', color=colors[count],
                markevery=10, markersize=8, linewidth=4,
                label='SPINN (t=%.1f)' % t
            )
            label = 'PyClaw' if i == 10 else None
            plt.plot(
                x_ex + 0.5, u_ex[i], '--',
                linewidth=6, color='black',
                label=label
            )
        plt.xlabel(r'$x$', fontsize=24)
        plt.ylabel(r'$u(x)$', fontsize=24)
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlim(0.0, 1.0)
        plt.ylim(-1.15, 1.5)
        fname = self.output_path(
            '%s_n%d.pdf' % (case.params['activation'], nodes)
        )
        plt.savefig(fname)
        plt.close()

    def run(self):
        self.make_output_dir()
        fname = os.path.join('code', 'data', 'pyclaw_burgers1d_sine2.npz')
        exact = np.load(fname)
        for case in self.cases:
            pth = sorted(Path(case.input_path()).glob('model_*.pt'))
            nn_state = torch.load(str(pth[0]))
            nodes = self._n_nodes(nn_state)
            self._plot_solution(case, nodes, exact)

        times = [0.1, 0.3, 0.6, 1.0]
        case = self.cases[1]
        data, models = get_results(case, times)
        plot_fd_centers(self, models, times)


class BurgersST(Problem):
    def get_name(self):
        return 'burgers_st'

    def _n_nodes(self, nn_state):
        return len(nn_state['layer1.x'])

    def setup(self):
        base_cmd = (
            'python3 code/burgers1d.py -d $output_dir --ic sin '
        )

        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n{i}'),
                base_command=base_cmd,
                nodes=i, samples=i*5,
                n_train=10000,
                lr=2e-3,
                tol=1e-3,
                duration=1.0,
                activation=activation,
                sample_frac=1.0,
                viscosity=0.0
            )
            for i in (50, 100, 200, 400) for activation in ('gaussian',)
        ]
        self.cases.extend([
            Simulation(
                root=self.input_path(f'{activation}_n_100'),
                base_command=base_cmd,
                nodes=100, samples=500,
                n_train=10000,
                lr=1e-3,
                activation=activation,
                kernel_size=5,
                sample_frac=1.0
            )
            for activation in ('kernel', 'nnkernel')
        ])

    def _plot_solution(self, case, nodes, exact):
        res = np.load(case.input_path('results.npz'))
        x, t = res['x'], res['t']
        u = res['u']
        colors = ['black', 'blue', 'green', 'violet']
        x_ex = exact['x']
        u_ex = exact['u']
        figure()

        for count, i in enumerate((0, 3, 6, 10)):
            plt.plot(
                x_ex, u_ex[i],
                linewidth=4, color=colors[count],
                label='Exact (t=%.1f)' % t[0, i]
            )
            label = 'SPINN' if i == 10 else None
            plt.plot(
                x[:, i], u[:, i], '--', color='red',
                linewidth=4, label=label
            )
        plt.xlabel(r'$x$', fontsize=24)
        plt.ylabel(r'$u(x)$', fontsize=24)
        plt.legend(loc='upper left')
        plt.grid()
        plt.xlim(-1.0, 1.0)
        plt.ylim(-0.25, 1.5)
        fname = self.output_path(
            '%s_n%d.pdf' % (case.params['activation'], nodes)
        )
        plt.savefig(fname)
        plt.close()

    def _plot_solution_nodes(self, case, fname):
        sd = np.load(case.input_path('results.npz'))
        state = torch.load(case.input_path('model.pt'))
        sdata = dict(x=sd['xp'], y=sd['tp'], u=sd['up'])
        plot_solution_nodes(sdata, state, fname)

    def run(self):
        self.make_output_dir()
        fname = os.path.join('code', 'data', 'pyclaw_burgers1d_sine.npz')
        exact = np.load(fname)
        for case in self.cases:
            nn_state = torch.load(case.input_path('model.pt'))
            nodes = self._n_nodes(nn_state)
            self._plot_solution(case, nodes, exact)
            self._plot_solution_nodes(
                case, self.output_path('%s_centers.png' % case.name)
            )


class TimeVarying(Problem):
    def _n_nodes(self, nn_state):
        return len(nn_state['layer1.center'])

    def _plot_fd_centers(self, models, times):
        plot_fd_centers(self, models, times)

    def _compare_solution(self, case, st_case, times):
        data, models = get_results(case, times)
        self._plot_fd_centers(models, times)
        sd = np.load(st_case.input_path('results.npz'))
        st_time = sd['t']
        t0 = st_time[0, 0]
        dt = st_time[0, 1] - t0
        figure()
        for i, t in enumerate(times):
            d = data[i]
            plt.plot(
                d['x'], d['y'], color='blue', marker='o', markersize=10,
                markevery=3, linewidth=4, label='FD (t=%.2f)' % t
            )
            j = round((t - t0)/dt)
            plt.plot(
                sd['x'][:, j], sd['u'][:, j], color='red', marker='^',
                markersize=10, markevery=3, linewidth=4,
                label='ST'
            )
            label = 'Exact' if i == (len(times) - 1) else None
            plt.plot(
                d['x'], d['y_exact'], '--', color='black',
                linewidth=6, label=label
            )
        plt.xlabel(r'$x$', fontsize=32)
        plt.ylabel(r'$u(x)$', fontsize=32)

    def _plot_solution(self, case, st_case, times):
        self._compare_solution(case, st_case, times)
        plt.legend(loc='upper left', fontsize=20)
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0.0, 2.5)
        fname = self.output_path('u_compare.pdf')
        plt.savefig(fname)
        plt.close()

    def _plot_3d(self, st_case, warp_scale=0.5, cb=None):
        sd = np.load(st_case.input_path('results.npz'))
        sd = dict(x=sd['xp'], y=sd['yp'], u=sd['up'], u_exact=sd['uex_p'])
        fname = self.output_path('st_sol.png')
        plot_3d_solution(sd, warp_scale)
        if cb:  # Callback to make some corrections
            cb()
        mlab.savefig(fname)
        mlab.close()

    def _plot_centers(self, st_case, xlabel='x', ylabel='y', cb=None):
        pth = Path(st_case.input_path('model.pt'))
        nn_state = torch.load(str(pth))
        x = nn_state['layer1.x']
        y = nn_state['layer1.y']
        w = nn_state['layer1.h']
        plot_centers(x, y, w)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(-0.25, 1.25)
        plt.axis('equal')
        plt.tight_layout()
        if cb:
            cb()
        fname = self.output_path('centers_st.pdf')
        plt.savefig(fname)
        plt.close()


class Heat(TimeVarying):
    def get_name(self):
        return 'heat'

    def setup(self):
        base_cmd = (
            'python3 code/heat1d_fd_spinn.py -d $output_dir '
        )

        self.fd_cases = [
            Simulation(
                root=self.input_path(f'fd_n{i}'),
                base_command=base_cmd,
                nodes=i, samples=200,
                n_train=5000,
                lr=1e-3,
                tol=4e-2,
                dt=0.0025,
                t_skip=4,
                duration=0.2,
                activation='gaussian',
                sample_frac=1.0,
            )
            for i in (20,) for activation in ('gaussian',)
        ]
        base_cmd = 'python3 code/heat1d.py -d $output_dir '
        self.st_cases = [
            Simulation(
                root=self.input_path(f'st_n100'),
                base_command=base_cmd,
                nodes=100, samples=400,
                n_train=10000,
                lr=1e-3,
                tol=5e-4,
                duration=0.5,
                activation='gaussian',
                sample_frac=1.0,
            )
        ]
        self.cases = self.fd_cases + self.st_cases

    def run(self):
        self.make_output_dir()
        case = self.fd_cases[0]
        st_case = self.st_cases[0]
        times = [0.01, 0.05, 0.1, 0.2]
        self._plot_solution(case, st_case, times)

        def _adjust():
            mlab.move(right=-0.15, up=-0.05)

        self._plot_3d(st_case, cb=_adjust)
        self._plot_centers(st_case, xlabel=r'$x$', ylabel=r'$t$')
        times.insert(0, 0.0)
        data, models = get_results(case, times)
        self._plot_fd_centers(models, times)


class Advection(TimeVarying):
    def get_name(self):
        return 'advection'

    def _n_nodes(self, nn_state):
        return len(nn_state['layer1.x'])

    def setup(self):
        base_cmd = 'python3 code/advection1d_fd_spinn.py -d $output_dir '

        self.fd_cases = [
            Simulation(
                root=self.input_path(f'fd_n10'),
                base_command=base_cmd,
                nodes=15, samples=100,
                n_train=5000,
                lr=5e-3,
                tol=1e-6,
                dt=0.0025,
                t_skip=40,
                duration=1.0,
            )
        ]

        base_cmd = (
            'python3 code/advection1d.py -d $output_dir '
        )

        self.st_cases = [
            Simulation(
                root=self.input_path(f'st_n40'),
                base_command=base_cmd,
                nodes=40, samples=800,
                duration=1.0,
                n_train=5000,
                lr=1e-3,
                activation='gaussian',
            )
        ]

        self.cases = self.fd_cases + self.st_cases

    def _plot_solution(self, case, st_case, times):
        self._compare_solution(case, st_case, times)
        plt.legend(loc='upper left', fontsize=20)
        plt.grid()
        plt.xlim(-1.0, 1.0)
        plt.ylim(-0.1, 1.25)
        fname = self.output_path('u_compare.pdf')
        plt.savefig(fname)
        plt.close()

    def _plot_solution_nodes(self, case):
        sd = np.load(case.input_path('results.npz'))
        state = torch.load(case.input_path('model.pt'))
        sdata = dict(x=sd['xp'], y=sd['yp'], u=sd['up'])
        fname = self.output_path('sol_centers.png')
        plot_solution_nodes(sdata, state, fname)

    def run(self):
        self.make_output_dir()
        case = self.fd_cases[0]
        st_case = self.st_cases[0]
        times = [0.1, 0.5, 1.0]
        self._plot_solution(case, st_case, times)
        self._plot_3d(st_case, warp_scale=1.0)
        self._plot_centers(st_case)
        self._plot_solution_nodes(st_case)


if __name__ == '__main__':
    PROBLEMS = [
        Misc,
        ODE1, ODE2, ODE3,
        ODE1Conv1, ODE1Conv3, ODE1Conv7,
        ODE3Conv1, ODE3Conv3,
        ODE2Conv5,
        ODE2Rep5,
        ODE1Var, ODE3Var,
        ODE1Fourier, ODE3Fourier,
        ODE3Comp,
        Poisson2DSineConv,
        Poisson2DSineNodes,
        SquareSlit,
        Irregular,
        Advection,
        BurgersFD,
        BurgersST,
        Heat,
        Cavity,
        CavityPySPH
    ]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()

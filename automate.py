#!/usr/bin/env python
import os
from re import L
from pathlib import Path

from automan.api import Problem
from automan.api import Automator, Simulation, filter_cases
import numpy as np
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from mayavi import mlab


fontsize=32
font = {'size': fontsize}
matplotlib.rc('font', **font)


def figure():
    plt.figure(figsize=(12, 12))
    font = {'size': 24}
    matplotlib.rc('font', **font)


def _plot_1d(problem, left_bdy=True, right_bdy=True):
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
            f"{problem.get_name()}_n_{case.params['nodes']}"
        ))
        plt.close()


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
                n_train=10000,
                lr=1e-4,
                tol=2.5e-5
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
                nodes=i, samples=20*i,
                sample_frac=0.1,
                n_train=50000,
                lr=2e-3,
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
                n_train=20000,
                lr=1e-3,
                tol=1e-3
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
    plt.tight_layout()
    plt.savefig(problem.output_path(
        f'{pname}_Linf_error_n_{n_nodes}.pdf'
    ))
    plt.close()


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
                n_train=20000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation=activation
            )
            for activation in ('gaussian', 'softplus', 'kernel')
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
                n_train=20000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation=activation
            )
            for activation in ('gaussian', 'softplus', 'kernel')
        ]

    def run(self):
        _plot_ode_conv(self, self.n, 'ode3')


def _plot_ode_conv_sampling(problem, n_nodes, 
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
        cen = [0.0] + nn_state['layer1.center'].tolist() + [1.0]
        plt.plot(
            cen, np.zeros_like(cen),
            'bo', markersize=8, label='Nodes'
        )

        plt.tight_layout()
        plt.savefig(problem.output_path(f"n_{n_nodes}_f_{case.params['sample_frac']}.pdf"))
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
    plt.tight_layout()
    plt.savefig(problem.output_path(f'L1_error_n_{n_nodes}_f.pdf'))
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
    plt.tight_layout()
    plt.savefig(problem.output_path(f'L2_error_n_{n_nodes}_f.pdf'))
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
    plt.tight_layout()
    plt.savefig(problem.output_path(f'Linf_error_n_{n_nodes}_f.pdf'))
    plt.close()


class ODE2Conv6(Problem):
    def get_name(self):
        return 'ode2_conv_6'

    def setup(self):
        self.n = 6
        self.ns = 20*self.n

        base_cmd = (
            'python3 code/ode2.py -d $output_dir'
        )
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{self.n}_f_{f}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                sample_frac=f,
                n_train=20000, n_skip=1,
                lr=5e-3,
                tol=1e-3,
                activation='gaussian'
            )
            for f in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
        ]

    def run(self):
        _plot_ode_conv_sampling(self, self.n)


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
        m_plt = mlab.surf(x, y, u, opacity=1.0)
        mlab.surf(x, y, u_exact, representation='wireframe')
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
        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n_{self.n}'),
                base_command=base_cmd,
                nodes=self.n, samples=self.ns,
                n_train=25000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation=activation,
                gpu=None
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
        m_plt = mlab.surf(x, y, u, opacity=1.0)
        mlab.surf(x, y, u_exact, representation='wireframe')
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
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{n}'),
                base_command=base_cmd,
                nodes=n, samples=4*n,
                n_train=20000, n_skip=1,
                lr=1e-3,
                tol=1e-3,
                activation='softplus',
                gpu=None
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
        self.cases = [
            Simulation(
                root=self.input_path(f'n_{n}'),
                base_command=base_cmd,
                nodes=n, samples=3*n,
                n_train=10000, n_skip=100,
                lr=1e-3,
                tol=1e-4,
                activation='softplus',
                gpu=None
            )
            for n in (25, 50, 100, 200, 500)
        ]

    def run(self):
        _plot_pde_conv_nodes_fem(self)


# Irregular domain
# python poisson2d_irreg_dom.py --plot --lr 1e-3 --gpu


class Advection(Problem):
    def get_name(self):
        return 'advection'

    def _n_nodes(self, nn_state):
        return len(nn_state['layer1.x'])

    def setup(self):
        base_cmd = (
            'python3 code/advection1d.py -d $output_dir '
        )

        self.cases = [
            Simulation(
                root=self.input_path(f'{activation}_n{i}'),
                base_command=base_cmd,
                nodes=i, samples=800,
                n_train=5000,
                lr=1e-3,
                activation=activation,
            )
            for i in (10, 20, 40) for activation in ('softplus', 'gaussian',)
        ]

    def _plot_centers(self, case, nn_state):
        x = nn_state['layer1.x']
        y = nn_state['layer1.y']
        w = nn_state['layer1.h']
        figure()
        a = plt.gca()
        circles = [
            plt.Circle((xi, yi), radius=wi, linewidth=1, fill=False,
                       color='blue')
            for xi, yi, wi in zip(x, y, w[:len(x)])
        ]
        c = matplotlib.collections.PatchCollection(
            circles, match_original=True
        )
        a.add_collection(c)
        plt.xlim(-1.5, 1.5)
        plt.axis('equal')
        plt.tight_layout()
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        fname = self.output_path(
            'centers_n_%s_%d.pdf' % (case.params['activation'], len(x))
        )
        plt.savefig(fname)
        plt.close()

    def _plot_solution(self, case, nodes):
        res = np.load(case.input_path('results.npz'))
        x, t = res['x'], res['t']
        u, u_ex = res['u'], res['u_exact']
        colors = ['black', 'blue', 'green']
        figure()

        for i in range(3):
            plt.plot(
                x[:, i], u_ex[:, i],
                linewidth=4, color=colors[i],
                label='Exact (t=%.1f)' % t[0, i]
            )
            label = 'SPINN' if i == 2 else None
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
            'n_%s_%d.pdf' % (case.params['activation'], nodes)
        )
        plt.savefig(fname)
        plt.close()

    def run(self):
        self.make_output_dir()
        for case in self.cases:
            nn_state = torch.load(case.input_path('model.pt'))
            nodes = self._n_nodes(nn_state)
            self._plot_solution(case, nodes)
            self._plot_centers(case, nn_state)


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
        figure()
        s1 = plt.subplot(211)
        s1.plot(ug, yg, 'o', color='red', label='Ghia et al.')
        s2 = plt.subplot(212)
        s2.plot(xg, vg, 'o', color='red', label='Ghia et al.')
        ls = [':', '--', '-']
        for i, case in enumerate(self.cases):
            nn_state = torch.load(case.input_path('model.pt'))
            nodes = self._n_nodes(nn_state)
            res = np.load(case.input_path('results.npz'))
            xc, uc, vc = res['xc'], res['uc'], res['vc']
            s1.plot(uc, xc, ls[i], lw=3,
                    label='SPINN (%d nodes)' % nodes)
            s2.plot(xc, vc, ls[i], lw=3,
                    label='SPINN (%d nodes)' % nodes)

        s1.set_xlabel(r'$u$')
        s1.set_ylabel(r'$y$')
        s1.legend()
        s2.set_xlabel(r'$x$')
        s2.set_ylabel(r'$v$')
        s2.legend()
        fname = self.output_path('centerline_compare.pdf')
        plt.savefig(fname)
        plt.close()

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


def get_results(case, times):
    files = sorted(Path(case.input_path()).glob('results_*.npz'))
    data = [np.load(str(f)) for f in files]
    result = []
    count = 0
    for time in times:
        while count < len(data):
            d = data[count]
            count += 1
            if abs(d['t'] - time) < 1e-10:
                result.append(d)
                break
    assert len(result) == len(times)
    return result


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
        data = get_results(case, [0.1, 0.3, 0.6, 1.0])
        colors = ['black', 'blue', 'green', 'violet']
        x_ex = exact['x']
        u_ex = exact['u']
        figure()

        for count, i in enumerate((1, 3, 6, 10)):
            t = i*0.1
            plt.plot(
                x_ex + 0.5, u_ex[i],
                linewidth=4, color=colors[count],
                label='Exact (t=%.1f)' % t
            )
            label = 'SPINN' if i == 10 else None
            res = data[count]
            plt.plot(
                res['x'], res['y'], '--', color='red',
                linewidth=4, label=label
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


if __name__ == '__main__':
    PROBLEMS = [
        ODE1, ODE2, ODE3,
        ODE3Conv1, ODE3Conv3,
        ODE2Conv6,
        Poisson2DSineConv,
        Poisson2DSineNodes,
        SquareSlit,
        Advection,
        BurgersFD,
        Cavity
    ]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()

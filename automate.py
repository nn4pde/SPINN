#!/usr/bin/env python
import os

from automan.api import Problem
from automan.api import Automator, Simulation, filter_cases
import numpy as np
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def _plot_1d(problem):
    problem.make_output_dir()
    for case in problem.cases:
        plt.figure(figsize=(12,12))
        font = {'size'   : 32}
        matplotlib.rc('font', **font)

        res = np.load(case.input_path('results.npz'))
        nn_state = torch.load(case.input_path('model.pt'))
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
        plt.xlabel(r'$x$', fontsize=32)
        plt.ylabel(r'$u(x)$', fontsize=32)
        plt.legend()
        plt.grid()

        cen = [0.0] + nn_state['layer1.center'].tolist() + [1.0]
        plt.plot(
            cen, np.zeros_like(cen), 
            'bo', markersize=8, label='Nodes'
        )
        
        plt.tight_layout()
        plt.savefig(problem.output_path('n_%d.pdf' % case.params['nodes']))
        plt.close()


class ODE1(Problem):
    def get_name(self):
        return 'ode1'

    def setup(self):
        base_cmd = (
            'python3 code/ode_basic.py -d $output_dir '
            '--de simple'
        )
        self.cases = [
            Simulation(
                root=self.input_path('n_%d' % i),
                base_command=base_cmd,
                nodes=i, samples=5*i,
                n_train=10000,
                lr=1e-4,
                activation='softplus',
                no_pu=None
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
            'python3 code/neumann1d.py -d $output_dir '
        )
        self.cases = [
            Simulation(
                root=self.input_path('n_%d' % i),
                base_command=base_cmd,
                nodes=i, samples=8*i,
                n_train=50000,
                lr=1e-4,
                activation='gaussian',
                no_pu=None
            )
            for i in (6, 10, 15)
        ]

    def run(self):
        _plot_1d(self)


class ODE3(Problem):
    def get_name(self):
        return 'ode3'

    def setup(self):
        base_cmd = (
            'python3 code/ode_basic.py -d $output_dir '
            '--de pulse'
        )
        self.cases = [
            Simulation(
                root=self.input_path('n_%d' % i),
                base_command=base_cmd,
                nodes=i, samples=5*i,
                n_train=20000,
                lr=1e-3,
                activation='softplus',
                no_pu=None
            )
            for i in (5, 9, 12)
        ]

    def run(self):
        _plot_1d(self)


if __name__ == '__main__':
    PROBLEMS = [ODE1, ODE2, ODE3]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()

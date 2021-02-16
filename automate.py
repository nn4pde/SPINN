#!/usr/bin/env python
import os

from automan.api import Problem
from automan.api import Automator, Simulation, filter_cases
import numpy as np
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

class ODE1(Problem):
    def get_name(self):
        return 'ode1'

    def setup(self):
        base_cmd = (
            'python3 code/spinn1d.py -d $output_dir '
            '--de simple'
        )
        self.cases = [
            Simulation(
                root=self.input_path('n_%d' % i),
                base_command=base_cmd,
                nodes=i, samples=5*i,
                n_train=10000,
                lr=1e-3,
                activation='softplus',
                no_pu=None
            )
            for i in (1, 3, 7)
        ]

    def run(self):
        self.make_output_dir()
        for case in self.cases:
            plt.figure()
            res = np.load(case.input_path('results.npz'))
            nn_state = torch.load(case.input_path('model.pt'))
            plt.plot(
                res['x'], res['y_exact'], label='Exact'
            )
            plt.plot(
                res['x'], res['y'], '--', label='SPINN'
            )
            plt.xlabel(r'$x$')
            plt.ylabel(r'$u(x)$')
            plt.legend()
            plt.grid()
            cen = [0.0] + nn_state['layer1.center'].tolist() + [1.0]
            plt.plot(cen, np.zeros_like(cen), 'o')
            plt.savefig(self.output_path('n_%d.pdf' % case.params['nodes']))
            plt.close()


class ODE2(ODE1):
    def get_name(self):
        return 'ode2'

    def setup(self):
        base_cmd = (
            'python3 code/neumann1d.py -d $output_dir '
            '-a gaussian'
        )
        self.cases = [
            Simulation(
                root=self.input_path('n_%d' % i),
                base_command=base_cmd,
                nodes=i, samples=i*8
            )
            for i in (6, 10, 15)
        ]


class ODE3(ODE1):
    def get_name(self):
        return 'ode3'

    def setup(self):
        base_cmd = (
            'python3 code/spinn1d.py -d $output_dir '
            '--de pulse -a softplus'
        )
        self.cases = [
            Simulation(
                root=self.input_path('n_%d' % i),
                base_command=base_cmd,
                nodes=i, samples=100
            )
            for i in (5, 9, 12)
        ]


if __name__ == '__main__':
    PROBLEMS = [ODE1, ODE2, ODE3]
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()

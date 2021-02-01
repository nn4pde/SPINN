#!/usr/bin/env python
import os

from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_cases
import numpy as np
import matplotlib
matplotlib.use('pdf')

# XXX

if __name__ == '__main__':
    PROBLEMS = []
    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()

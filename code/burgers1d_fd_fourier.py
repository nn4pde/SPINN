# Hybrid Finite Difference Fourier-SPINN algorithm 
# for the 1d Burgers equation

import numpy as np
import torch

from fourier1d_base import Fourier1D
from fd_spinn1d_base import AppFD1D, FDPlotter1D
from burgers1d_fd_spinn import Burgers1D

class FourierPlotter(FDPlotter1D):
    def plot_weights(self):
        pass

if __name__ == '__main__':
    app = AppFD1D(
        pde_cls=Burgers1D,
        nn_cls=Fourier1D,
        plotter_cls=FourierPlotter
    )
    app.run(modes=20, lr=1e-3, dt=0.01)

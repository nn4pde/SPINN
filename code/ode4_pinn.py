'''PINN version of ode4.py
'''

from ode4 import ODE4, App1D
from pinn1d_base import PINN1D, PINNPlotter1D

if __name__ == '__main__':
    app = App1D(
        pde_cls=ODE4,
        nn_cls=PINN1D,
        plotter_cls=PINNPlotter1D
    )
    app.run(layers=5, neurons=200, samples=800, n_train=20000,
            lr=5e-4, tol=1e-3)

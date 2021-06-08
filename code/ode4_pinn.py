'''PINN version of ode4.py
'''

from ode4 import ODE4, App1D, tensor
from pinn1d_base import PINN1D, PINNPlotter1D


class ODE4PINN(ODE4):
    def boundary_loss(self, nn):
        u = nn(self.boundary())
        ub = tensor(self.exact(self.xbn))
        bc = u - ub
        return (bc**2).sum()


if __name__ == '__main__':
    app = App1D(
        pde_cls=ODE4PINN,
        nn_cls=PINN1D,
        plotter_cls=PINNPlotter1D
    )
    app.run(layers=5, neurons=200, samples=2000, n_train=20000,
            lr=5e-4, tol=1e-3, sample_frac=0.25)

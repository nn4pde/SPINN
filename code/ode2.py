import numpy as np

import torch
import torch.autograd as ag

from common import tensor
from spinn1d import App1D, SPINN1D, Plotter1D
from ode_base import BasicODE


class Neumann1D(BasicODE):
    def fixed_nodes(self):
        return np.array([0.0])

    def pde(self, x, u, ux, uxx):
        return uxx + np.pi*np.pi*u - np.pi*torch.sin(np.pi*x)

    def has_exact(self):
        return True

    def exact(self, x):
        return -0.5*x*np.cos(np.pi*x)

    def interior_loss(self, nn):
        res = self._get_residue(nn)
        return (res**2).sum()

    def boundary_loss(self, nn):
        x = self.boundary()
        x.requires_grad = True
        u = nn(x)
        du = ag.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            retain_graph=True
        )
        ub = tensor([0.0, 0.5])
        dbc = (u - ub)[:1]
        nbc = (du[0] - ub)[1:]
        bc = torch.cat((dbc, nbc))
        return 50*(bc**2).sum()

    def plot_points(self):
        n = 25
        x = np.linspace(0.0, 1.0, n)
        return x


if __name__ == '__main__':
    app = App1D(
        pde_cls=Neumann1D, nn_cls=SPINN1D,
        plotter_cls=Plotter1D
    )
    app.run(
        nodes=5, samples=100, sample_frac=0.1, n_train=50000,
        lr=2e-3, tol=1e-3
    )

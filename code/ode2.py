import numpy as np

import torch
import torch.autograd as ag

from common import tensor
from spinn1d import App1D, SPINN1D, Plotter1D
from ode_base import BasicODE


class Neumann1D(BasicODE):
    def __init__(self, n, ns):
        super().__init__(n, ns)
        self.xb.requires_grad = True

    def eval_bc(self, problem):
        x = self.boundary()
        u = problem.nn(x)
        du = ag.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            retain_graph=True
        )
        ub = tensor([0.0, 0.5])
        dbc = (u - ub)[:1]
        nbc = (du[0] - ub)[1:]
        return torch.cat((dbc, nbc))

    def pde(self, x, u, ux, uxx):
        return uxx + np.pi*np.pi*u - np.pi*torch.sin(np.pi*x)
    
    def has_exact(self):
            return True

    def exact(self, x):
        return -0.5*x*np.cos(np.pi*x)

    def boundary_loss(self, nn):
        x = self.boundary()
        u = nn(x)
        du = ag.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            retain_graph=True
        )
        ub = tensor([0.0, 0.5])
        dbc = (u - ub)[:1]
        nbc = (du[0] - ub)[1:]
        bc = torch.cat((dbc, nbc))
        return 100*(bc**2).sum()


if __name__ == '__main__':
    app = App1D(
        pde_cls=Neumann1D, nn_cls=SPINN1D,
        plotter_cls=Plotter1D
    )
    app.run(nodes=20, samples=80, lr=1e-2)

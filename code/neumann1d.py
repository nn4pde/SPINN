import numpy as np
import torch
from spinn1d import RegularDomain, App1D, SPINN1D, Problem1D, tensor
import torch.autograd as ag


class MyDomain(RegularDomain):
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
        return uxx + np.pi*(np.pi*u - torch.sin(np.pi*x))

    def exact(self, x):
        return -0.5*x*np.cos(np.pi*x)


if __name__ == '__main__':
    app = App1D(Problem1D, SPINN1D, MyDomain)
    app.run(nodes=20, samples=80, lr=1e-2)

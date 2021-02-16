import numpy as np
import torch
from oned import RegularDomain, App1D, tensor
from spinn1d import SPINN1D, SPINNProblem1D
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
        ub = tensor(list(map(problem.bc, x)))
        dbc = (u - ub)[:1]
        nbc = (du[0] - ub)[1:]
        return torch.cat((dbc, nbc))


class MyProblem(SPINNProblem1D):
    @classmethod
    def from_args(cls, domain, nn, args):
        return cls(domain, nn, None)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        pass

    def pde(self, x, u, ux, uxx):
        return uxx + np.pi*(np.pi*u - torch.sin(np.pi*x))

    def bc(self, x):
        tol = 1e-4

        if abs(x) < tol:
            return 0.0
        elif abs(x - 1.0) < tol:
            return 0.5

    def has_exact(self):
        return True

    def exact(self, x):
        return -0.5*x*np.cos(np.pi*x)

    def show_exact(self):
        return True


if __name__ == '__main__':
    app = App1D(MyProblem, SPINN1D, MyDomain)
    app.run(nodes=20, samples=80, lr=1e-2)

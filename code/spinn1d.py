'''SPINN in 1D with nodes that move and have either variable or fixed widths.
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from oned import Problem1D, tensor, App1D


class Shift(nn.Module):
    def __init__(self, points, fixed_points, fixed_h=False):
        super().__init__()
        n_free = len(points)
        n_fixed = len(fixed_points)
        self.n = n = n_free + n_fixed
        if n_fixed:
            xmin = min(np.min(points), np.min(fixed_points))
            xmax = max(np.max(points), np.max(fixed_points))
        else:
            xmin, xmax = np.min(points), np.max(points)
        dx = (xmax - xmin)/n
        self.center = nn.Parameter(tensor(points))
        self.fixed = tensor(fixed_points)
        if fixed_h:
            self.h = nn.Parameter(torch.tensor(2.0*dx))
        else:
            self.h = nn.Parameter(2.0*dx*torch.ones(n))

    def centers(self):
        return torch.cat((self.center, self.fixed))

    def forward(self, x):
        cen = self.centers()
        return (x - cen)/self.h


class SPINN1D(nn.Module):
    @classmethod
    def from_args(cls, domain, activation, args):
        return cls(domain, activation,
                   fixed_h=args.fixed_h, use_pu=not args.no_pu)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--fixed-h', dest='fixed_h', action='store_true', default=False,
            help='Use fixed width nodes.'
        )
        p.add_argument(
            '--no-pu', dest='no_pu', action='store_true', default=False,
            help='Do not use a partition of unity.'
        )

    def __init__(self, domain, activation, fixed_h=False, use_pu=True):
        super().__init__()

        self.fixed_h = fixed_h
        self.use_pu = use_pu
        self.layer1 = Shift(domain.nodes(), domain.fixed_nodes(),
                            fixed_h=fixed_h)
        n = self.layer1.n
        self.activation = activation
        self.layer2 = nn.Linear(n, 1, bias=not use_pu)
        self.layer2.weight.data.fill_(0.0)
        if not self.use_pu:
            self.layer2.bias.data.fill_(0.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.layer1(x)
        y = self.activation(y)
        if self.use_pu:
            y1 = y.sum(axis=1).unsqueeze(1)
        else:
            y1 = tensor(1.0)
        y = self.layer2(y/y1)
        return y.squeeze()

    def centers(self):
        return self.layer1.centers()

    def widths(self):
        return self.layer1.h

    def weights(self):
        return self.layer2.weight

    def show(self):
        print("Basis centers: ", self.centers())
        print("Mesh widths: ", self.layer1.h)
        print("Nodal weights: ", self.layer2.weight)
        print("Output bias: ", self.layer2.bias)


class SPINNProblem1D(Problem1D):
    def plot_weights(self):
        x = self.nn.centers().detach().cpu().numpy()
        w = self.nn.weights().detach().cpu().squeeze().numpy()
        if not self.plt2:
            self.plt2, = plt.plot(x, np.zeros_like(x), 'o', label='centers')
            self.plt3, = plt.plot(x, w, 'o', label='weights')
        else:
            self.plt2.set_data(x, np.zeros_like(x))
            self.plt3.set_data(x, w)


if __name__ == '__main__':
    from oned import RegularDomain
    app = App1D(
        problem_cls=SPINNProblem1D, nn_cls=SPINN1D,
        domain_cls=RegularDomain
    )
    app.run(nodes=20, samples=80, lr=1e-2)

'''SPINN in 1D with nodes that move and have either variable or fixed widths.
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from oned import Case1D, main


class Shift(nn.Module):
    def __init__(self, n, fixed_h=False):
        super().__init__()
        self.n = n
        dx = 1.0/(n + 1)
        xl, xr = 0.0, 1.0
        self.center = nn.Parameter(torch.linspace(xl+dx, xr - dx, n))
        if fixed_h:
            self.h = nn.Parameter(torch.tensor(2.0*dx))
        else:
            self.h = nn.Parameter(2.0*dx*torch.ones_like(self.center))

    def forward(self, x):
        return (x - self.center)/self.h


class SPINN1D(nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.activation,
                   fixed_h=args.fixed_h, use_pu=not args.no_pu)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--nodes', '-n', dest='nodes',
            default=kw.get('nodes', 10), type=int,
            help='Number of nodes to use.'
        )
        p.add_argument(
            '--fixed-h', dest='fixed_h', action='store_true', default=False,
            help='Use fixed width nodes.'
        )
        p.add_argument(
            '--no-pu', dest='no_pu', action='store_true', default=False,
            help='Do not use a partition of unity.'
        )

    def __init__(self, n, activation, fixed_h=False, use_pu=True):
        super().__init__()

        self.fixed_h = fixed_h
        self.use_pu = use_pu
        self.n = n
        self.activation = activation
        self.layer1 = Shift(n, fixed_h=fixed_h)
        self.layer2 = nn.Linear(n, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.layer1(x)
        y = self.activation(y)
        if self.use_pu:
            y1 = y.sum(axis=1).unsqueeze(1)
        else:
            y1 = 1.0
        y = self.layer2(y)
        return y/y1

    def show(self):
        params = list(self.parameters())
        print("Basis centers: ", params[0])
        print("Mesh widths: ", params[1])
        print("Nodal weights: ", params[2])
        print("Output bias: ", params[3])


class SPINNCase1D(Case1D):
    def plot_weights(self):
        x = self.nn.layer1.center.detach().squeeze().numpy()
        w = list(self.nn.layer2.parameters())[0].detach().squeeze().numpy()
        if not self.plt2:
            self.plt2, = plt.plot(x, np.zeros_like(x), 'o', label='centers')
            self.plt3, = plt.plot(x, w, 'o', label='weights')
        else:
            self.plt2.set_data(x, np.zeros_like(x))
            self.plt3.set_data(x, w)


if __name__ == '__main__':
    main(SPINN1D, SPINNCase1D, nodes=20, samples=80, lr=1e-2)

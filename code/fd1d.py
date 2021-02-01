'''A simple finite difference neural network.

Our approach is to use fixed width kernels like a meshless method except with
the particles all fixed on a grid. We then set a fixed stencil to evaluate the
kernels. This gives us very good performance for large number of points.

'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from oned import Case1D, main, setup_argparse


class FDNet(nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.activation, args.n_offset)

    def __init__(self, n, activation, n_offset=3):
        super().__init__()
        dx = 1.0/(n - 1)
        self.dx = dx
        no = n_offset
        self.n = n
        self.activation = activation
        self.noffset = no
        self.ksize = self.noffset*2 + 1
        self.np = n + 2*self.noffset
        self.u = nn.Parameter(torch.zeros(self.np))
        self.x = torch.linspace(-dx*no, 1 + dx*no, self.np)

    def forward(self, x):
        no = self.noffset
        h = self.dx
        idx = torch.empty(len(x), dtype=int)
        idx[:] = x//h
        factor = 1.0/np.sqrt(np.pi)
        y = torch.zeros_like(x)
        for i in range(0, 2*no + 1):
            ix = idx + i
            diff = (x - self.x[ix])/h
            y += self.activation(diff)*self.u[ix]
        return y*factor


class FDCase(Case1D):
    def plot_weights(self):
        x = self.nn.x
        if not self.plt2:
            plt2, = plt.plot(x, np.zeros_like(x), '.')
            self.plt2 = plt2
            plt.legend()
        else:
            self.plt2.set_data(x, np.zeros_like(x))


def setup(**kw):
    p = setup_argparse(**kw)
    p.add_argument(
        '--n-offset', dest='n_offset', type=int, default=3,
        help='Size of the kernel on each side of a point.'
    )
    return p


if __name__ == '__main__':
    p = setup(nodes=20, samples=60, lr=1e-2)
    main(FDNet, FDCase, p)

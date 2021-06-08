# Base classes for implementing Fourier SPINN

import numpy as np
import torch.nn as nn

from common import tensor
from spinn1d import Plotter1D


class PINN1D(nn.Module):
    @classmethod
    def from_args(cls, pde, activation, args):
        return cls(pde, activation, args.neurons, args.layers, args.method)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--neurons', dest='neurons', type=int,
            default=kw.get('neurons', 10),
            help='Number of neurons per layer.'
        )
        p.add_argument(
            '--layers', dest='layers', default=kw.get('layers', 5),
            type=int, help='Number of hidden layers.'
        )
        p.add_argument(
            '--method', dest='method', default=kw.get('method', 'generic'),
            action='store', choices=['kernel', 'simple', 'generic'],
            help='The method to use to setup the neurons and layers.'
        )

    def __init__(self, pde, activation, neurons=10, layers=5,
                 method='generic'):
        super().__init__()
        # PDE and activation are ignored and not used here.
        self.n = n = neurons
        self.nl = nl = layers
        self.method = method

        # SPINN with Kernel network
        if method == 'generic':
            # Generic
            layers = [nn.Linear(1, n), nn.Tanh()]
            for i in range(nl - 2):
                layers.extend([nn.Linear(n, n), nn.Tanh()])
            layers.append(nn.Linear(n, 1))
        else:
            raise ValueError('Invalid method %s.' % method)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.model(x).squeeze()

    def centers(self):
        return tensor([]), tensor([])

    def widths(self):
        return tensor([])


class PINNPlotter1D(Plotter1D):
    def plot_weights(self):
        pass

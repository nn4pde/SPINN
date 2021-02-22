# Base classes for implementing Fourier SPINN

import numpy as np
import torch
import torch.nn as nn

from common import tensor
from spinn1d import Plotter1D


class Scale1D(nn.Module):
    def __init__(self, n_modes):
        super().__init__()
        self.n_modes = n_modes

    def forward(self, x):
        ws = np.zeros((1, self.n_modes))
        
        for i in range(self.n_modes):
            ws[0,i] = i + 1.0

        ws = tensor(ws)
        return x @ ws


## Implementation of Fourier network
## Note: The arguments 'pde', 'activation' are dummy
##       They are used to ensure compatibility with the SPINN codes
class Fourier1D(nn.Module):
    @classmethod
    def from_args(cls, pde, activation, args):
        return cls(pde, activation, args.modes)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--modes', '-m', dest='modes', 
            default=kw.get('modes', 10), type=int,
            help='Number of Fourier modes.'
        )

    def __init__(self, pde, activation, n_modes=10):
        super().__init__()

        self.layer1 = Scale1D(n_modes)
        self.layer2 = nn.Linear(2*n_modes, 1)

        self.layer2.weight.data.fill_(0.0)
        self.layer2.bias.data.fill_(0.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat((torch.sin(self.layer1(x)), torch.cos(self.layer1(x))), 1)
        x = self.layer2(x)
        return x.squeeze()


class FourierPlotter1D(Plotter1D):
    def plot_weights(self):
        pass
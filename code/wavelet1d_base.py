# Base classes for implementing Wavelet SPINN

import numpy as np
import torch
import torch.nn as nn

from common import tensor
from spinn1d import Plotter1D, Kernel


# Mother wavelets
def mexicanhat(x):
    sigma = 1.0
    factor = 2.0/(np.sqrt(3*sigma)*(np.pi**0.25))
    return factor*(1.0 - (x/sigma)**2)*torch.exp(-x*x/(2.0*sigma*sigma))
    

class ShiftScale1D(nn.Module):
    def __init__(self, wavelet, n_shift, n_scale):
        super().__init__()
        self.wavelet = wavelet
        self.n_shift = n_shift
        self.n_scale = n_scale
        self.n_wavelet = (2*self.n_shift + 1)*(2*self.n_scale + 1)
        I, J = np.mgrid[-self.n_shift:(self.n_shift + 1), -self.n_scale:(self.n_scale + 1)]
        self.I, self.J = tensor(I).unsqueeze(2), tensor(J).unsqueeze(2)
        self.F1 = 2.0**(0.5*self.J)
        self.F2 = 2.0**self.J

    def forward(self, x):
        phis = self.F1*self.wavelet(
            self.F2 @ x.squeeze().unsqueeze(0) - self.I
        )
        return phis.reshape(self.n_wavelet, x.shape[0]).T


## Implementation of Wavelet network
## Note: The arguments 'pde', 'activation' are dummy
##       They are used to ensure compatibility with the SPINN codes
class Wavelet1D(nn.Module):
    @classmethod
    def from_args(cls, pde, activation, args):
        return cls(pde, activation, 
            args.wavelet, args.n_scale, args.n_shift)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--wavelet', '-w', dest='wavelet',
            default=kw.get('wavelet', 'mexicanhat'),
            choices=['mexicanhat'],
            help='Mother wavelet for the wavelet decomposition.'
        )
        p.add_argument(
            '--n-scale', dest='n_scale', 
            default=kw.get('n_scale', 5), type=int,
            help='Number of levels of scaling in wavelet transform.'
        )
        p.add_argument(
            '--n-shift', dest='n_shift', 
            default=kw.get('n_shift', 5), type=int,
            help='Number of levels of shifting in wavelet transform.'
        )
        
    def _get_wavelet(self, wavelet):
        wavelets = {
            'mexicanhat': lambda : mexicanhat
        }
        return wavelets[wavelet]()

    def __init__(self, pde, activation,
        wavelet='mexicanhat', n_scale=5, n_shift=5):
        super().__init__()

        self.wavelet = wavelet
        self.f_wavelet = self._get_wavelet(wavelet)
        self.n_scale = n_scale
        self.n_shift = n_shift

        self.n_wavelet = (2*n_scale + 1)*(2*n_shift + 1)

        self.layer1 = ShiftScale1D(self.f_wavelet, n_scale, n_shift)
        self.layer2 = nn.Linear(self.n_wavelet, 1)

        self.layer2.weight.data.fill_(0.0)
        self.layer2.bias.data.fill_(0.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x.squeeze()


class WaveletPlotter1D(Plotter1D):
    def plot_weights(self):
        pass
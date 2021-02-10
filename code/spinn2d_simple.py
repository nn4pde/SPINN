import numpy as np
import torch.nn as nn

from twod import Case2D, Shift2D, main


class SPINN2D(nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.activation,
                   fixed_h=args.fixed_h, use_pu=not args.no_pu)

    @classmethod
    def setup_argparse(cls, parser, **kw):
        p = parser
        p.add_argument(
            '--nodes', '-n', dest='nodes',
            default=kw.get('nodes', 25), type=int,
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

    def __init__(self, n_nodes, activation, fixed_h=False, use_pu=True):
        super().__init__()

        self.activation = activation
        self.use_pu = use_pu
        n = round(np.sqrt(n_nodes) + 0.49)
        points = np.mgrid[0:1:n*1j, 0:1:n*1j]
        self.n_nodes = n*n
        self.layer1 = Shift2D(points, fixed_h=fixed_h)
        self.layer2 = nn.Linear(self.n_nodes, 1, bias=not use_pu)
        self.layer2.weight.data.fill_(0.0)
        if not self.use_pu:
            self.layer2.bias.data.fill_(0.0)

    def forward(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        xh, yh = self.layer1(x, y)
        z = self.activation(xh, yh)
        if self.use_pu:
            zsum = z.sum(axis=1).unsqueeze(1)
        else:
            zsum = 1.0
        z = self.layer2(z)
        return (z/zsum).squeeze()


if __name__ == '__main__':
    main(SPINN2D, Case2D, nodes=40, samples=120, lr=1e-2)

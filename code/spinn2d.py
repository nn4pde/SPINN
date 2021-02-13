import torch.nn as nn

from twod import RegularDomain, App2D, Problem2D, Shift2D


class SPINN2D(nn.Module):
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

        self.activation = activation
        self.use_pu = use_pu
        self.layer1 = Shift2D(domain.nodes(), domain.fixed_nodes(),
                              fixed_h=fixed_h)
        n = self.layer1.n
        self.layer2 = nn.Linear(n, 1, bias=not use_pu)
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
        z = self.layer2(z/zsum)
        return z.squeeze()

    def centers(self):
        return self.layer1.centers()

    def widths(self):
        return self.layer1.widths()

    def weights(self):
        return self.layer2.weight


if __name__ == '__main__':
    app = App2D(
        problem_cls=Problem2D, nn_cls=SPINN2D,
        domain_cls=RegularDomain
    )
    app.run(nodes=40, samples=120, lr=1e-2)

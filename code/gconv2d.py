import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn

from common import tensor
from spinn2d import App2D, Plotter2D
from poisson2d_sine import Poisson2D


class SPHConv(MessagePassing):
    def __init__(self, activation):
        super().__init__(aggr='add', flow='target_to_source', node_dim=0)
        self.activation = activation

    def forward(self, x, y, pos, h, u, edge_index):
        xn = pos[:, 0]
        yn = pos[:, 1]
        return self.propagate(
            edge_index, h=h, u=u, x=x, y=y, xn=xn, yn=yn
        )

    def message(self, xn_j, yn_j, h_j, u_j, x_i, y_i):
        xh = (x_i - xn_j)/h_j
        yh = (y_i - yn_j)/h_j
        return u_j*self.activation(xh, yh)


class GConvNet(nn.Module):
    @classmethod
    def from_args(cls, pde, activation, args):
        return cls(pde, activation, fixed_h=args.fixed_h,
                   use_pu=not args.no_pu, max_nbrs=args.max_nbrs)

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
        p.add_argument(
            '--max-nbrs', dest='max_nbrs', type=int,
            default=kw.get('max_nbrs', 49),
            help='Maximum number of neighbors to use.'
        )

    def __init__(self, pde, activation,
                 fixed_h=False, use_pu=True, max_nbrs=25):
        super().__init__()
        self.activation = activation
        self.use_pu = use_pu
        self.max_nbrs = max_nbrs
        points = pde.nodes()
        fixed_points = pde.fixed_nodes()
        n_free = len(points[0])
        n_fixed = len(fixed_points[0])
        self.n = n = n_free + n_fixed
        if n_fixed:
            dsize = np.ptp(np.hstack((points[0], fixed_points[0])))
        else:
            dsize = np.ptp(points[0])
        dx = dsize/np.sqrt(n)
        self.dx = dx

        pts = tensor(np.zeros((n_free, 2)))
        pts[:, 0] = tensor(points[0])
        pts[:, 1] = tensor(points[1])
        self.points = nn.Parameter(pts)
        fpts = tensor(np.zeros((n_fixed, 2)))
        fpts[:, 0] = tensor(fixed_points[0])
        fpts[:, 1] = tensor(fixed_points[1])
        self.f_points = fpts
        self.fixed_h = fixed_h
        if fixed_h:
            self.h = nn.Parameter(tensor(dx))
        else:
            self.h = nn.Parameter(dx*tensor(np.ones(self.n)))

        self.u = nn.Parameter(tensor(np.zeros(self.n)))
        self.sph = SPHConv(self.activation)

    def forward(self, x, y):
        target = torch.stack((x, y), dim=1)
        nodes = torch.vstack((self.points, self.f_points))
        a_index = knn(nodes, target, self.max_nbrs)
        h = self.widths()
        if self.use_pu:
            dnr = self.sph.forward(x, y, nodes,
                                   h, 1.0, a_index)
        else:
            dnr = 1.0
        nr = self.sph.forward(x, y, nodes,
                              h, self.u, a_index)
        return nr/dnr

    def centers(self):
        nodes = torch.vstack((self.points, self.f_points))
        return nodes.T

    def widths(self):
        if self.fixed_h:
            return self.h*torch.ones_like(self.u)
        else:
            return self.h

    def weights(self):
        return self.u


if __name__ == '__main__':
    app = App2D(
        pde_cls=Poisson2D, nn_cls=GConvNet,
        plotter_cls=Plotter2D
    )
    app.run(nodes=40, samples=120, lr=1e-2)

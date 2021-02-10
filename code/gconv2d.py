import numpy as np
from mayavi import mlab
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn

from twod import Case2D, main, tensor


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


class SPHNet(nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args.nodes, args.activation, fixed_h=args.fixed_h,
                   use_pu=not args.no_pu, max_nbrs=args.max_nbrs)

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
        p.add_argument(
            '--max-nbrs', dest='max_nbrs', type=int,
            default=kw.get('max_nbrs', 25),
            help='Maximum number of neighbors to use.'
        )

    def __init__(self, n_nodes, activation,
                 fixed_h=False, use_pu=True, max_nbrs=25):
        super().__init__()
        self.activation = activation
        self.use_pu = use_pu
        self.max_nbrs = max_nbrs
        n = round(np.sqrt(n_nodes) + 0.49)
        x, y = np.mgrid[0:1:n*1j, 0:1:n*1j]
        self.n_nodes = n_nodes = n*n
        self.dx = dx = 1.0/n

        x = x.ravel()
        y = y.ravel()
        pts = tensor(np.zeros((self.n_nodes, 2)))
        pts[:, 0] = tensor(x)
        pts[:, 1] = tensor(y)
        self.points = nn.Parameter(pts)
        if fixed_h:
            self.h = nn.Parameter(tensor(dx))
        else:
            self.h = nn.Parameter(dx*tensor(np.ones(self.n_nodes)))

        self.u = nn.Parameter(tensor(np.zeros(self.n_nodes)))
        self.sph = SPHConv(self.activation)

    def forward(self, x, y):
        target = torch.stack((x, y), dim=1)
        a_index = knn(self.points, target, self.max_nbrs)
        if self.use_pu:
            dnr = self.sph.forward(x, y, self.points,
                                   self.h, 1.0, a_index)
        else:
            dnr = 1.0
        nr = self.sph.forward(x, y, self.points,
                              self.h, self.u, a_index)
        return nr/dnr


class ConvCase(Case2D):

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        nn = self.nn
        x = nn.points[:, 0].detach().numpy()
        y = nn.points[:, 1].detach().numpy()
        if not self.plt2:
            self.plt2 = mlab.points3d(
                x, y, np.zeros_like(x), scale_factor=0.025
            )
        else:
            self.plt2.mlab_source.trait_set(x=x, y=y)


if __name__ == '__main__':
    main(SPHNet, ConvCase, nodes=40, samples=120, lr=1e-2)

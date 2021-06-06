import os
from mayavi import mlab
import numpy as np
import torch
import torch.autograd as ag

from common import tensor
from spinn2d import Plotter2D, App2D, SPINN2D
from pde2d_base import RegularPDE


class CavityPDE(RegularPDE):
    def __init__(self, n_nodes, ns, nb=None, nbs=None, sample_frac=1.0):
        self.sample_frac = sample_frac

        # Interior nodes
        n = round(np.sqrt(n_nodes) + 0.49)
        self.n = n
        dxb2 = 1.0/(n + 1)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, n*1j)
        x, y = np.mgrid[sl, sl]
        self.i_nodes = (x.ravel(), y.ravel())

        # Fixed nodes
        nb = n if nb is None else nb
        self.nb = nb
        if nb == 0:
            self.f_nodes = ([], [])
        else:
            dxb2 = 1.0/(nb)
            _x = np.linspace(dxb2, 1.0 - dxb2, nb)
            _o = np.ones_like(_x)
            x = np.hstack((_x, _o, _x, 0.0*_o))
            y = np.hstack((_o*0.0, _x, _o, _x))
            self.f_nodes = (x, y)

        # Interior samples
        self.ns = ns = round(np.sqrt(ns) + 0.49)
        dxb2 = 1.0/(ns)
        xl, xr = dxb2, 1.0 - dxb2
        sl = slice(xl, xr, ns*1j)
        x, y = np.mgrid[sl, sl]
        xs, ys = (tensor(t.ravel(), requires_grad=True)
                  for t in (x, y))
        self.p_samples = (xs, ys)

        # Boundary samples
        nbs = ns if nbs is None else nbs
        self.nbs = nbs
        sl = slice(0, 1.0, nbs*1j)
        _x = np.linspace(dxb2, 1.0 - dxb2, nbs)
        o = np.ones_like(_x)

        def tg(x):
            return tensor(x, requires_grad=True)

        self.left = left = (tg(0.0*o), tg(_x))
        self.right = right = (tg(o), tg(_x))
        self.bottom = bottom = (tg(_x), tg(o)*0.0)
        self.top = top = (tg(_x), tg(o))
        self.b_samples = (
            torch.cat([x[0] for x in (top, bottom, left, right)]),
            torch.cat([x[1] for x in (top, bottom, left, right)])
        )

        self.n_interior = len(self.p_samples[0])
        self.rng_interior = np.arange(self.n_interior)
        self.sample_size = int(self.sample_frac*self.n_interior)

    def _compute_gradient(self, u, xs, ys):
        du = ag.grad(
            outputs=u, inputs=(xs, ys), grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )
        return du[0], du[1]

    def n_vars(self):
        return 3

    def has_exact(self):
        return False

    def interior_loss(self, nn):
        xs, ys = self.interior()
        U = nn(xs, ys)
        u = U[:, 0]
        v = U[:, 1]
        p = U[:, 2]
        u, ux, uy, uxx, uyy = self._compute_derivatives(u, xs, ys)
        v, vx, vy, vxx, vyy = self._compute_derivatives(v, xs, ys)
        px, py = self._compute_gradient(p, xs, ys)

        # The NS equations.
        nu = 0.01
        ce = ((ux + vy)**2).sum()
        mex = ((u*ux + v*uy + px - nu*(uxx + uyy))**2).sum()
        mey = ((u*vx + v*vy + py - nu*(vxx + vyy))**2).sum()

        return ce + (mex + mey)

    def boundary_loss(self, nn):
        bc_weight = self.ns*20
        xb, yb = self.boundary()
        Ub = nn(xb, yb)
        ub = Ub[:, 0]
        vb = Ub[:, 1]
        pb = Ub[:, 2]
        pbx, pby = self._compute_gradient(pb, xb, yb)

        n_top = len(self.top[0])
        ub_top = ub[:n_top]
        bc_loss = ((ub_top - 1.0)**2).sum()
        bc_loss += (ub[n_top:]**2).sum()*bc_weight
        bc_loss += ((vb)**2).sum()*bc_weight
        bc_loss += (pby[:2*n_top]**2).sum()
        bc_loss += (pbx[2*n_top:]**2).sum()
        return bc_loss

    def plot_points(self):
        n = min(self.ns*2, 200)
        x, y = np.mgrid[0:1:n*1j, 0:1:n*1j]
        return x, y


class CavityPlotter(Plotter2D):

    def get_plot_data(self):
        x, y = self.pde.plot_points()
        xt, yt = tensor(x.ravel()), tensor(y.ravel())
        pn = self.nn(xt, yt).detach().cpu().numpy()
        pn.shape = x.shape + (3,)
        return x, y, pn

    def plot_weights(self):
        '''Implement this method to plot any weights.
        Note this is always called *after* plot_solution.
        '''
        nn = self.nn
        x, y = nn.centers()
        h = nn.widths().detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        if not self.plt2:
            self.plt2 = mlab.points3d(
                x, y, np.zeros_like(x), h, mode='2dcircle',
                scale_factor=1.0, color=(1, 0, 0), opacity=0.4
            )
            self.plt2.glyph.glyph_source.glyph_source.resolution = 20
            mlab.pipeline.glyph(
                self.plt2, color=(1, 0, 0), scale_factor=0.025, opacity=0.4,
                mode='2dcross', scale_mode='none'
            )
        else:
            self.plt2.mlab_source.trait_set(x=x, y=y, scalars=h)

    def plot_solution(self):
        xn, yn, pn = self.get_plot_data()
        u, v, p = pn[..., 0], pn[..., 1], pn[..., 2]
        for arr in (xn, yn, u, v, p):
            arr.shape = arr.shape + (1, )
        vmag = np.sqrt(u*u + v*v)
        pde = self.pde
        if self.plt1 is None:
            mlab.figure(
                size=(700, 700), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1)
            )
            if self.show_exact and pde.has_exact():
                un = pde.exact(xn, yn)
                mlab.surf(xn, yn, un, colormap='viridis',
                          representation='wireframe')
            src = mlab.pipeline.vector_field(
                xn, yn, np.zeros_like(xn), u, v, np.zeros_like(u),
                scalars=vmag, name='vectors'
            )
            self.plt1 = mlab.pipeline.vectors(
                src, scale_factor=0.2, mask_points=3, colormap='viridis'
            )
            self.plt1.scene.z_plus_view()
            cgp = mlab.pipeline.contour_grid_plane(
                src, opacity=0.8,
                colormap='viridis',
            )
            cgp.enable_contours = False
            mlab.colorbar(self.plt1)
        else:
            self.plt1.mlab_source.trait_set(u=u, v=v, scalars=vmag)
        return self.get_error(xn, yn, pn)

    def save(self, dirname):
        modelfname = os.path.join(dirname, 'model.pt')
        torch.save(self.nn.state_dict(), modelfname)
        rfile = os.path.join(dirname, 'results.npz')
        x, y, u = self.get_plot_data()
        lc = tensor(np.linspace(0, 1, 100))
        midc = tensor(0.5*np.ones(100))
        xc = torch.cat((lc, midc))
        yc = torch.cat((midc, lc))
        data = self.nn(xc, yc).detach().cpu().numpy()
        vc = data[:, 1][:100]
        uc = data[:, 0][100:]

        np.savez(rfile, x=x, y=y, u=u, xc=lc, uc=uc, vc=vc)


if __name__ == '__main__':
    app = App2D(
        pde_cls=CavityPDE, nn_cls=SPINN2D,
        plotter_cls=CavityPlotter,
    )
    app.run(nodes=100, samples=2500, sample_frac=0.1, lr=5e-4, n_train=5000)

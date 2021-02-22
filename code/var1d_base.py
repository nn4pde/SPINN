# Base class for variational implementation of SPINN

from ode_base import BasicODE

class Var1D(BasicODE):
    # This will now store the variational form of the PDE
    def pde(x, u, ux, uxx):
        pass

    def interior_loss(self, nn):
        res = self._get_residue(nn)
        return res.mean()

'''Fourier version of ode4.py
'''

if __name__ == '__main__':
    from ode4 import ODE4, App1D
    from fourier1d_base import Fourier1D, FourierPlotter1D
    app = App1D(
        pde_cls=ODE4,
        nn_cls=Fourier1D,
        plotter_cls=FourierPlotter1D
    )
    app.run(modes=200, samples=800, n_train=20000,
            lr=5e-4, tol=1e-3)

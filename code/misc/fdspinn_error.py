import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat


def err_fd_burgers_sin2():
    d = np.load('../../outputs/burgers_fd/sin2_gaussian_n40/results_0100.npz')
    e = np.load('../data/pyclaw_burgers1d_sine2.npz')
    # Shifting is needed since this is from -0.5, 0.5 and ours from 0, 1
    ie = interp1d(e['x'] + 0.5, e['u'][10])
    ic = interp1d(d['x'], d['y'])
    x = np.linspace(0, 1, 250)
    print("Burgers FD Sin2 L1 error:", np.mean(np.abs(ie(x) - ic(x))))


def err_fd_burgers_sin():
    d = np.load('../../outputs/burgers_fd/sin_gaussian_n150/results_0200.npz')
    e = np.load('../data/pyclaw_burgers1d_sine.npz')
    ie = interp1d(e['x'], e['u'][10])
    ic = interp1d(d['x'], d['y'])
    x = np.linspace(min(e['x']), max(e['x']), 500)
    print("Burgers FD Sin L1 error:", np.mean(np.abs(ie(x) - ic(x))))


def err_st_burgers_sin2():
    d = np.load('../../outputs/burgers_st/sin2_gaussian_n_400/results.npz')
    e = np.load('../data/pyclaw_burgers1d_sine2.npz')
    ie = interp1d(e['x'], e['u'][10])
    ic = interp1d(d['x'][:,10], d['u'][:,10])
    x = np.linspace(min(e['x']), max(e['x']), 500)
    print("Burgers ST Sin2 L1 error:", np.mean(np.abs(ie(x) - ic(x))))


def err_st_burgers_sin():
    d = np.load('../../outputs/burgers_st/sin_gaussian_n400/results.npz')
    e = np.load('../data/pyclaw_burgers1d_sine.npz')
    ie = interp1d(e['x'], e['u'][10])
    ic = interp1d(d['x'][:,10], d['u'][:,10])
    x = np.linspace(min(e['x']), max(e['x']), 500)
    print("Burgers ST Sin L1 error:", np.mean(np.abs(ie(x) - ic(x))))


def err_allen_cahn():
    d = np.load('../../outputs/allen_cahn/n_100/results_0180.npz')
    assert np.abs(d['t']  - 0.9) < 1e-12
    e = loadmat('../data/AC.mat')
    dt = e['tt'][0, 1]
    x = e['x'][0]
    index = int(0.9/dt)
    ie = interp1d(x, e['uu'][:, index])
    ic = interp1d(d['x'], d['y'])
    x = np.linspace(min(e['x']), max(e['x']), 500)
    print("Allen Cahn L1 error:", np.mean(np.abs(ie(x) - ic(x))))


def main():
    err_fd_burgers_sin2()
    err_fd_burgers_sin()
    err_st_burgers_sin2()
    err_st_burgers_sin()
    err_allen_cahn()

if __name__ == '__main__':
    main()

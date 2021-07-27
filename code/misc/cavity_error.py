import sys
import os

import numpy as np
from scipy.interpolate import interp1d


fname = os.path.join('..', 'data', 'ghia_re_100.txt')
xg, vg, yg, ug = np.loadtxt(fname, unpack=True)


def _get_error(pth, kind='spinn'):
    fn = os.path.join(pth, 'results.npz')
    res = np.load(fn)
    if kind == 'spinn':
        x = res['xc']
        u = res['uc']
        v = res['vc']
    else:
        x = res['x']
        u = res['u_c']
        v = res['v_c']

    fu = interp1d(x, u)
    fv = interp1d(x, v)

    # The plots are u vs y and v vs x.

    ud = np.abs(fu(yg) - ug)
    vd = np.abs(fv(xg) - vg)
    return np.max(ud), np.max(vd)


def get_error_spinn(pth):
    return _get_error(pth, kind='spinn')


def get_error_pysph(pth):
    return _get_error(pth, kind='pysph')


if __name__ == '__main__':
    spinn_dirs = ['../../outputs/cavity/n100', '../../outputs/cavity/n200',
                  '../../outputs/cavity/n300']
    pysph_dirs = ['../../outputs/cavity_pysph/nx_50']
    u, v = [], []
    labels = ['SPINN 100', 'SPINN 200', 'SPINN 300', 'PySPH 50x50']
    for pth in spinn_dirs:
        ue, ve = get_error_spinn(pth)
        u.append(ue)
        v.append(ve)
    for pth in pysph_dirs:
        ue, ve = get_error_pysph(pth)
        u.append(ue)
        v.append(ve)
    import pandas as pd
    df = pd.DataFrame(dict(name=labels, linf_u=u, linf_v=v))
    print(df)
    print(df.to_latex(index=False))

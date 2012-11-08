from motorlab.tuning.gam_new import compress_late_dims, \
    get_mean_sem_0th_dim, \
    gam_predict_cv
from motorlab.datasets import datasets
from motorlab.datacollection import DataCollection

from amcmorl_py_tools.curves import gauss1d

import motorlab.kinematics as kin
import numpy as np
from scipy.stats import pearsonr

def test_compress_late_dims():
    a = np.array([[[0, 0],
                   [2, 4]],
                  [[2, 1],
                   [0, 4]],
                  [[1, 4],
                   [3, 0]]])
    b = compress_late_dims(a)
    c = np.array([[0, 0, 2, 4],
                  [2, 1, 0, 4],
                  [1, 4, 3, 0]])
    np.testing.assert_array_equal(b,c)

def test_get_mean_sem_0th_dim():
    a = np.array([[[1., 2., 3., 1],
                   [1., 1., 1., 0],
                   [1., np.nan, 0., 0.]],
                  [[1., 1., 1., np.nan],
                   [0., np.nan, 0., 0.],
                   [1., 1.,  0., 0.]]])
    
    b_mean, b_sem, b_count = get_mean_sem_0th_dim(a)
    c_mean = np.array([11/11., 5/10.])
    c_count = np.array([11, 10])
    np.testing.assert_array_almost_equal(b_mean, c_mean)
    np.testing.assert_array_equal(b_count, c_count)

def test_gam_predict_cv():
    dsname = 'frank-osmd'
    align = 'hold'
    lag = 0.1 # s
    b0 = 20 # Hz
    noise_sd = 1. # Hz
    b_scale = 10.
    nbin = 10

    ds = datasets[dsname]
    dc = DataCollection(ds.get_files()[:5])
    unit = ds.get_units()[0]
    dc.add_unit(unit, lag)
    bnd = dc.make_binned(nbin=nbin, align=align, do_count=True)
    ntask, nrep, nedge, ndim = bnd.pos.shape
    
    pos = bnd.pos.reshape(ntask * nrep, nedge, -1)
    drn = kin.get_dir(pos)
    
    # simplest model
    # y = b0 + Bd.D
    B = np.array([.2, .6, .4]) * b_scale
    y = b0 + np.dot(drn, B)
    
    noise = np.random.normal(0, noise_sd, size=y.shape)
    y += noise
    
    bnd.set_count_from_flat(y[:,None])
    out = gam_predict_cv(bnd, ['kd'], [dsname, unit, lag, align], \
        family='gaussian')
    
    have = np.mean(out.coef[0], axis=0)[:4]
    want = np.array([b0, B[0], B[1], B[2]])

    np.testing.assert_array_almost_equal(have, want, decimal=1)

    # changing Bdt model, gaussian noise model
    # y = b0 + Bdt.D
    gc_type = [('A', float), ('fwhm', float), ('c', float)]
    gcoeff = np.array([[(1., 1.5, 3.), (.75, 2.5, 6.)],
                       [(.4, 4.,  2.), (.1,  2.,  5.)],
                       [(.8, 3.,  5.), (.6, 4.5, 6.5)]], dtype=gc_type)
    gcoeff = gcoeff.view(np.recarray)

    x  = np.linspace(0, 10, nbin)
    Bt = np.zeros((3, nbin))

    for Bkt, gc in zip(Bt, gcoeff):
        g0 = gauss1d(x, gc.A[0], gc.fwhm[0], gc.c[0])
        g1 = gauss1d(x, gc.A[1], gc.fwhm[1], gc.c[1])
        Bkt[:] = g0 + g1
        
    Bt *= b_scale
    y = b0 + np.sum(Bt.T[None] * drn, axis=-1)
    y = np.random.normal(y, noise_sd)
    bnd.set_count_from_flat(y[:,None])
    out = gam_predict_cv(bnd, ['kdX'], [dsname, unit, lag, align], \
        family='gaussian')
        
    # check that actual and predicted are correlated
    x = out.actual.flatten()
    y = out.pred[8].flatten()
    mc, rp = pearsonr(x,y)
    np.testing.assert_approx_equal(mc, 1, significant=1)

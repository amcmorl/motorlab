# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import amcmorl_py_tools.vecgeom.stats as ss
from amcmorl_py_tools.vecgeom import unitvec, norm
from warnings import warn
from motorlab.tuning import fit

def bootstrap_pd_stats(b, cov, neural_data, model, ndim=3):
    '''
    Parameters
    ----------
    b : ndarray
      coefficients
    cov : ndarray
      covariance matrix
    neural_data : ndarray
      counts (GLM) or rates (OLS)
    model : string
      specification of fit model

    Returns
    -------
    pd : ndarray
      preferred directions
    ca : ndarray
      confidence angle of preferred direction
    kd : ndarray
      modulation depths
    kdse : ndarray
      standard errors of modulation depths
    '''
    # number of independent samples from original data
    d = 'd' if 'd' in model else 'v'
    nsamp = np.sum(~np.isnan(neural_data.sum(axis=1)))
    # compressing along last axis gives number of samples per bin
    # which is correct, since bootstrapping is done independently
    # for each bin
    
    # bootstrap a distribution of b values
    # using mean, covariance matrix from GLM (/ OLS).
    bootb = np.random.multivariate_normal(b, cov, (nsamp,))
    bootdict = fit.unpack_many_coefficients(bootb, model, ndim=ndim)
    if 'X' in model:
        bootpd = unitvec(bootdict[d], axis=2)
        # has shape nsamp, nbin, ndim
    else:
        bootpd = unitvec(bootdict[d], axis=1)
        # has shape nsamp, ndim

    # get mean pd to narrow kappa estimation
    bdict = fit.unpack_coefficients(b, model, ndim=ndim)
    if 'X' in model:
        nbin = bdict[d].shape[0]
        pd = unitvec(bdict[d], axis=1)
        pdca = np.zeros((nbin))
        for i in xrange(nbin):
            # estimate kappa
            k = ss.estimate_kappa(bootpd[:,i], mu=pd[i])
            # estimate ca (preferred direction Confidence Angle)
            pdca[i] = ss.measure_percentile_angle_ex_kappa(k)
            
        # calculate the standard error of the bootstrapped PDs
        kd = norm(bootdict[d], axis=2)
        kdse = np.std(kd, axis=0, ddof=1)
    else:
        nbin = 1
        pd = unitvec(bdict[d])
        k = ss.estimate_kappa(bootpd, mu=pd)
        pdca = ss.measure_percentile_angle_ex_kappa(k)
        bootkd = norm(bootdict[d], axis=1)
        kd = np.mean(bootkd, axis=0)
        kdse = np.std(bootkd, axis=0, ddof=1)
    return pd, pdca, kd, kdse

def mean_deviation(pds):
    '''
    Calculate the largest angle between any two vectors in pds.

    Parameters
    ----------
    pds : array_like, shape (n_dirs, 3)
      preferred directions

    Returns
    -------
    max_ang : float
      maximum angular deviation between any two pds
    '''
    assert np.rank(pds) == 2
    assert pds.shape[1] == 3
    eps = 1e-8
    dots = np.dot(pds, pds.T)
    difnt = ~np.eye(dots.shape[0], dtype=bool)
    nearones = np.abs(dots - 1.) < eps
    dots[nearones] = 1.
    return stats.nanmean(np.arccos(dots[difnt]))
    
def max_deviation(pds):
    '''
    Calculate the largest angle between any two vectors in pds.

    Parameters
    ----------
    pds : array_like, shape (n_dirs, 3)
      preferred directions

    Returns
    -------
    max_ang : float
      maximum angular deviation between any two pds
    '''
    assert np.rank(pds) == 2
    assert pds.shape[1] == 3

    if type(pds) == np.ma.MaskedArray:
        # propagate mask into all affected cells
        pds_mask = pds.mask.astype(int) + 1
        mask_dot = np.dot(pds_mask, pds_mask.T)
        dot_mask = mask_dot > np.min(mask_dot)
        pds = np.asarray(pds)
        dot = np.dot(pds, pds.T)
        dot_masked = dot[~dot_mask]
        angs = np.arccos(dot_masked)
        res = np.nanmax(angs)
        ind = np.nanargmax(angs)
        inds = np.unravel_index(ind, dot.shape)
        print "Maxes at %d and %d of %d" % (inds[0], inds[1], dot.shape[0])
        return res
    else:
        eps = 1e-8
        dots = np.dot(pds, pds.T)
        nearones = np.abs(dots - 1.) < eps
        dots[nearones] = 1.
        return np.nanmax(np.arccos(dots))

def calc_all_pd_uncerts(count, bds, bsed, ndim=3):
    '''
    Calculate uncertainties in PD estimates from directional regression
    coefficients and their standard errors.

    Parameters
    ----------
    count : array_like
      spike counts (PSTHs), shape (n_trials, n_bins)
    bds : array_like
      coefficients from regression, shape (n_calcs, n_coefs)
    bsed : array_like
      standard errors of b coefficients, shape (n_calcs, n_coefs)

    Returns
    -------
    pd : array_like
      preferred directions, shape (n_windows, 3)
    k : array_like
      modulation depths of pds, shape (n_windows)
    k_se : array_like
      standard deviations of modulation depths
    theta_percentile : array_like
      95% percentile angle of pd, shape (n_windows)

    Notes
    -----
    Number of samples should be calculated from number of non-zero elements
    in regression. With the moving PD calculations, time-changing coefficients
    are regressed using 0 values many times, but these aren't really
    observations. Should be okay as it is - i.e. number of non-nan trials,
    since each non-nan trial contributes one time point to each estimate of PD

    2011-Feb-01 - I think this should just be non-nan elements, since I have now
    switched to using GLM and counts rather than rates, and zero-counts are just
    one observation of the Poisson process: implementation remains the same.
    '''
    warn(DeprecationWarning("Doesn't calculate using covariance matrix. "
                            "Use ___ instead."))
    assert type(bds) == np.ndarray
    assert type(bsed) == np.ndarray

    n_samp = np.sum(~np.isnan(count.sum(axis=1)))
    n_calcs = bds.shape[0]

    pd = np.empty((n_calcs, ndim))
    k = np.empty((n_calcs))
    k_se = np.empty((n_calcs))       # std err of modulation depth
    kappa = np.empty((n_calcs))      # spread of pd Fisher dist
    R = np.empty((n_calcs))          # resultants of pd Fisher dist
    theta_pc = np.empty((n_calcs))   # 95% percentile angle

    for i in xrange(n_calcs):
        # calc pds from regression coefficients
        b_d = bds[i]
        k[i] = norm(b_d)
        pd[i] = b_d / k[i]
        k_se[i], kappa[i], R[i] = \
            old_bootstrap_pd_stats(b_d, bsed[i], k[i], n_samp)
        theta_pc[i] = ss.measure_percentile_angle_ex_kappa(kappa[i])
    return pd, k, k_se, theta_pc

def old_bootstrap_pd_stats(b, b_s, k, n_samp=int(1e3)):
    '''
    Calculate, by bootstrapping, statistics of preferred directions.

    Parameters
    ----------
    pd : array_like, shape (3)
      preferred directions
    b_s : float
      standard error of lin regress coefficients
    k : float
      modulation depth
    n_samp : int
      number of samples to use for bootstrap

    Returns
    -------
    k_s : array_like
      standard errors of k, modulation depth, shape (?)
    kappa : array_like
      dispersion factors, shape (?)
    R : array_like
      length factor of pd distribution ??, shape (?)

    Notes
    -----
    1. Angles of preferred direction cones are determined as follows.
    2. Regression coefficients (`b`) are recovered from pds and modulation
       depths.
    3. Random populations of regression coefficients are constructed from
       distributions with the same mean and standard deviation as estimated
       from the regression procedure, with the same number of samples as the
       original data.
    4. Preferred directions are calculated from these coefficient values
       (i.e. dividing by modulation depth).
    
    '''
    warn(DeprecationWarning("Doesn't calculate using covariance matrix. "
                            "Use ___ instead."))
    assert (type(n_samp) == int) | (type(n_samp) == np.int_)
    assert (type(k) == np.float_) or (type(k) == float)
    assert b.shape == b_s.shape == (3,)

    #pd = np.asarray(pd)

    # reconstruct regression coefficients
    # Now generate n_samp samples from normal populations
    # (mean: b_k, sd: err_k). Will have shape (n_samp, 3).
    b_rnd = np.random.standard_normal(size=(n_samp,3))
    b_rnd *= b_s
    b_rnd += b
    pd = unitvec(b)
    #io.savemat('bootstrap_b.mat', dict(b_rnd=b_rnd))
    ks = norm(b_rnd, axis=1)
    k_s = np.std(ks, ddof=1)
    pds_rnd = b_rnd / ks[...,np.newaxis]
    kappa = ss.estimate_kappa(pds_rnd, mu=pd)
    R, S = ss.calc_R(pds_rnd)
    return k_s, kappa, R

def max_neighbour_deviation(pds):
    '''
    Returns the maximum angular deviation between any two neighbouring pds

    Parameters
    ----------
    pds : array_like, shape (n, 3)
      list of 3-vectors

    Returns
    -------
    maximum angle : scalar
      maximum angle in radians between any two neighbours in pds
    '''

    assert np.rank(pds) == 2
    assert pds.shape[1] == 3

    dot = np.dot(pds, pds.T)
    neighbours = np.roll(np.eye(pds.shape[0]).astype(bool), 1, axis=1)
    angs = np.arccos(dot[neighbours][:-1])
    res = np.nanmax(angs)
    return res

def test_mean_deviation():
    # test simple case - all valid angles
    B = np.array([[ 0.26097211, -1.76208686,  0.46576422],
                  [-1.67251254,  1.0682131, -0.12919601],
                  [-0.17027653,  0.66272588, -0.59754741]])
    pd = unitvec(B, axis=1)
    angles = np.zeros((B.shape[0],))
    angles[0] = np.arccos(np.dot(pd[0], pd[1]))
    angles[1] = np.arccos(np.dot(pd[0], pd[2]))
    angles[2] = np.arccos(np.dot(pd[1], pd[2]))
    np.testing.assert_almost_equal(np.mean(angles), mean_deviation(pd))

import numpy as np
#from multiregress import multiregress, rsq_from_b
import statsmodels.api as sm
from motorlab.kinematics import get_vel, get_dir

nonan = lambda x: not np.any(np.isnan(x))

def glm_any_model(count, pos, time, model='', family='poisson-log'):
    '''
    Parameters
    ----------
    count : array_like
      spike counts, shape (n_trials, n_bins)
    pos : array_like
      positional data, shape (n_trials, n_bins + 1, n_dims)
    time : array_lie
      bin edge times, shape (n_trials, n_bins + 1)
    model : string
      model
    family : GLM distribution

    Returns
    -------
    b : ndarray
      coefficients
    cov : ndarray
      covariance matrix of b
    '''
    assert(family in ['poisson-log'])
    assert(nonan(pos))
    assert(nonan(time))
    endog, exog, offset = prepare_regressors(count, pos, time, model)
    loglink = sm.families.links.log
    fam = sm.families.Poisson(loglink)
    glmodel = sm.GLM(endog, exog, offset=offset, family=fam)
    results = glmodel.fit()
    b = results.params
    cov = results.normalized_cov_params
    # bse is diagonal of sqrt(cov)
    return b, cov

def ols_any_model(rate, pos, time, model=''):
    '''
    Parameters
    ----------
    rate : array_like
      spike rates, shape (n_trials, n_bins)
    pos : array_like
      positional data, shape (n_trials, n_bins + 1, n_dims)
    time : array_like
      bin edge times, shape (n_trials, n_bins + 1)
    model : string
      model

    Returns
    -------
    b : ndarray
      coefficients
    cov : ndarray
      covariance matrix of b
    '''
    assert(nonan(pos))
    assert(nonan(time))
    endog, exog, offset = prepare_regressors(rate, pos, time, model)
    ols = sm.OLS(endog, exog)
    results = ols.fit()
    b = results.params
    cov = results.normalized_cov_params * np.var(results.resid, ddof=1)
    # the above makes diagonal(cov) very close to results.bse
    
    rsq = results.rsquared
    # bse is diagonal of sqrt(cov)
    return b, cov, rsq

def unpack_coefficients(b, model, ndim=3):
    '''
    Converts (n,) coefficient vector into a dictionary of labelled
    coefficients, folded correctly by time.
    
    Parameters
    ----------
    b : ndarray
      shape (n,), coefficients from GLM/OLS
    model : string
      specification of fit model
    ndim : integer, optional
      number of spatial dimensions, default is 3
      
    Returns
    -------
    bdict : dictionary
      labelled, folded coefficients from fit
    '''
    # recover coefficients from regression product
    bdict = {}
    idx = 0
    last_idx = None
    if 'k' in model:
        bdict['k'] = b[-1]
        last_idx = -1
    if 'p' in model:
        bdict['p'] = b[idx:idx + ndim]
        idx += ndim
    if 's' in model:
        bdict['s'] = b[idx]
        idx += 1
    if ('v' in model) or ('d' in model):
        if 'd' in model:
            dest = 'd'
        elif 'v' in model:
            dest = 'v'
        if not 'X' in model:
            bdict[dest] = b[idx:last_idx]
        else:
            b_dir = b[idx:last_idx]
            bdict[dest] = b_dir.reshape(b_dir.shape[0] / ndim, ndim)
    return bdict
   
def unpack_many_coefficients(b, model, ndim=3):
    '''
    Separate coefficients from list returned by glm_any_model,
    based on `model`.
    
    Parameters
    ----------
    b : ndarray
      shape (n,k), n repeats of k coefficients
    model : string
      specification of glm model used
    '''
    nsamp = b.shape[0]
    bdict = {}
    
    # do one to get basic shape info
    tmp = unpack_coefficients(b[0], model, ndim=ndim)
    for k, v in tmp.iteritems():
        many_shape = [b.shape[0],] + list(v.shape)
        bdict[k] = np.zeros(many_shape)
        bdict[k][0] = v
    
    # now do all the rest
    for i in xrange(1, nsamp):
        tmp = unpack_coefficients(b[i], model, ndim=ndim)
        for k, v in tmp.iteritems():
            bdict[k][i] = v
    
    return bdict
   
def prepare_regressors(count, pos, time, model=''):
    '''
    Parameters
    ----------
    count : array_like
      firing rates, shape (ntrial, nbin)
    pos : array_like
      positional data, shape (ntrial, nbin + 1, ndim)
      these are positions at bin edges
    time : array_like
      bin edge times, shape (ntrial, nbin + 1)
    model : string
      model

    Returns
    -------
    reg_vars : ndarray

    reg_frs : ndarray
    '''
    count = np.asarray(count)
    assert(np.rank(count) == 2)
    ntrial, nbin = count.shape
    assert(not np.any(np.isnan(count)))
    pos = np.asarray(pos)
    assert(np.rank(pos) == 3)
    assert(pos.shape[0:2] == (ntrial, nbin + 1))
    time = np.asarray(time)
    assert(np.rank(time) == 2)
    assert(time.shape == (ntrial, nbin + 1))
    assert(type(model) == str)
                      
    #ntrial, nbin = count.shape
    ndim = pos.shape[-1]
    nvar = ntrial * nbin
    endog = count.reshape(nvar) # should be equivalent to flatten
    vels = None
    exog = None

    # add the time-invariant variables
    if 'p' in model:
        pos_flat = pos[:,1:,:].reshape(nvar, ndim)
        exog = add_column(exog, pos_flat)
    if 's' in model:
        if vels == None:
            vels = get_vel(pos, time, tax=1, spax=2)
        spds = np.sqrt(np.sum(vels**2, axis=2))
        spds_flat = spds.reshape(nvar)
        exog = add_column(exog, spds_flat)

    # do the potentially time-variant variables
    if ('d' in model or 'v' in model):
        if 'd' in model:
            var = get_dir(pos, tax=1, spax=2)
        elif 'v' in model:
            if vels == None:
                vels = get_vel(pos, time, tax=1, spax=2)
            var = vels
        if not 'X' in model: # time-static variable
            var_flat = np.reshape(var, (nvar, ndim))
        else: # time-dynamic variable
            mask = np.eye(nbin)
            var_flat = np.reshape(mask[None,...,None] * var[...,None,:],
                                  (nvar, nbin * ndim))
        exog = add_column(exog, var_flat)

    # do the constant + dynamic direction model
    if 'q' in model:
        assert ('X' in model)
        # for simplicity's sake, let us call this model 'kqX'
        assert ('d' not in model)
        assert ('v' not in model)

    if 'k' in model:
        exog = sm.tools.add_constant(exog, prepend=False)

    offset = np.log(np.diff(time, axis=-1).reshape(nvar))

    return endog, exog, offset

#~ def regress_any_model(frs, pos, time, model=''):
    #~ '''
    #~ Regresses firing rates and kinematics according to a given model.
    #~ 
    #~ Parameters
    #~ ----------
    #~ frs : array_like, shape (n_trials, n_bins)
      #~ firing rates
    #~ pos : array_like, shape (n_trials, n_bins, n_dims)
      #~ positional data
    #~ model : string
      #~ model
#~ 
    #~ Returns
    #~ -------
    #~ b : dict
      #~ dictionary of coefficients of regression
    #~ b_se : ndarray
      #~ standard errors of regression coefficients
      #~ might need to be divided up into dictionary too
    #~ r : ndarray
      #~ r-squared of fit
    #~ '''
    #~ assert(nonan(pos))
    #~ assert(nonan(time))
    #~ endog, exog, offset = prepare_regressors(frs, pos, time, model)
    #~ 
    #~ b, bse = multiregress(exog, endog, add_const=False, has_const=True)
    #~ # recover coefficients from regression product    
    #~ rsq = rsq_from_b(exog[:,:-1], endog, b[:-1])
    #~ 
    #~ b_named = {}
    #~ bse_named = {}
    #~ next_idx = 0
    #~ next_se_idx = 0
    #~ if 'k' in model:
        #~ b_named['k'] = b[next_idx]
        #~ next_idx += 1
    #~ if 'p' in model:
        #~ b_named['p'] = b[next_idx:next_idx + 3]
        #~ bse_named['p'] = bse[next_se_idx:next_se_idx + 3]
        #~ next_idx += 3
        #~ next_se_idx += 3
    #~ if 's' in model:
        #~ b_named['s'] = b[next_idx]
        #~ bse_named['s'] = bse[next_se_idx]
        #~ next_idx += 1
        #~ next_se_idx += 1
    #~ if ('v' in model) or ('d' in model): 
        #~ if not 'X' in model:
            #~ if 'd' in model:
                #~ b_named['d'] = b[next_idx:]
                #~ bse_named['d'] = bse[next_se_idx:]
            #~ elif 'v' in model:
                #~ b_named['v'] = b[next_idx:]
                #~ bse_named['v'] = bse[next_se_idx:]
        #~ else:
            #~ b_dir = b[next_idx:]
            #~ bse_dir = bse[next_se_idx:]
            #~ if 'd' in model:
                #~ b_named['d'] = b_dir.reshape(b_dir.shape[0]/3,3)
                #~ bse_named['d'] = bse_dir.reshape(bse_dir.shape[0]/3,3)
            #~ elif 'v' in model:
                #~ b_named['v'] = b_dir.reshape(b_dir.shape[0]/3,3)
                #~ bse_named['v'] = bse_dir.reshape(bse_dir.shape[0]/3,3)
    #~ return b_named, bse_named, rsq

def add_column(cur_vars, more_vars):
    if np.rank(more_vars) == 1:
        more_vars.shape = list(more_vars.shape) + [1,]
    if cur_vars != None: # must have same number of tasks
        assert np.rank(more_vars) == 2
        assert cur_vars.shape[0] == more_vars.shape[0]
        return np.concatenate((cur_vars, more_vars), axis=-1)
    else:
        return more_vars

#def regress_full_model_other(frs, pos, time, model=''):
#    '''
#    Regresses firing rates and kinematics according to a given model.
#    
#    Parameters
#    ----------
#    frs : array_like, shape (n_trials, n_bins)
#      firing rates
#    pos : array_like, shape (n_trials, n_bins, n_dims)
#      positional data
#    model : string
#      model
#
#    Returns
#    -------
#    b : dict
#      dictionary of coefficients of regression
#    b_se : ndarray
#      standard errors of regression coefficients
#      might need to be divided up into dictionary too
#    r : ndarray
#      r-squared of fit
#    '''
#    frs = np.asarray(frs)
#    assert(np.rank(frs) == 2)
#    pos = np.asarray(pos)
#    assert(np.rank(pos) == 3)
#    assert frs.shape[0:2] == pos.shape[0:2]
#    time = np.asarray(time)
#    assert(np.rank(time) == 2)
#    assert(type(model) == str)
#    assert(not np.any(np.isnan(frs)))
#    
#    n_trials, n_bins = frs.shape
#    n_bins -= 1 # because of -1 for velocity calculation (offset = 1 wrt frs)
#    n_dims = pos.shape[-1]
#    n_reg = n_trials * n_bins
#    reg_frs = frs[:,1:].reshape(n_reg)
#    vels = None
#    reg_vars = None
#
#    # add the time-invariant variables
#    if 'k' in model:
#        add_const = True
#    else:
#        add_const = False
#    if 'p' in model:
#        pos_flat = pos[:,1:,:].reshape(n_reg, 3)
#        reg_vars = add_column(reg_vars, pos_flat)
#    if 's' in model:
#        if vels == None:
#            vels = get_vel(pos, time, tax=1, spax=2)
#        spds = np.sqrt(np.sum(vels**2, axis=2))
#        spds_flat = spds.reshape(n_reg)
#        reg_vars = add_column(reg_vars, spds_flat)
#
#    # do the potentially time-variant variables
#    if ('d' in model or 'v' in model):
#        if (vels == None):
#            vels = get_vel(pos, time, tax=1, spax=2)
#        if 'd' in model:
#            speeds = np.sqrt(np.sum(vels**2, axis=2))
#            var = vels / speeds[...,None]
#        elif 'v' in model:
#            var = vels
#        if not 'X' in model: # time-static variable
#            var_flat = np.reshape(var, (n_reg, 3))
#        else: # time-dynamic variable
#            mask = np.eye(n_bins)
#            var_flat = np.reshape(mask[None,...,None] * var[...,None,:],
#                                  (n_reg, n_bins * n_dims))
#        reg_vars = add_column(reg_vars, var_flat)
#
#    # do the regression
#    b, b_se = multiregress(reg_vars, reg_frs, add_const=add_const)
#    
#    # recover coefficients from regression product
#    rsq = rsq_from_b(reg_vars, reg_frs, b)
#    b_named = {}
#    b_se_named = {}
#    next_idx = 0
#    next_se_idx = 0
#    if 'k' in model:
#        b_named['k'] = b[next_idx]
#        next_idx += 1
#    if 'p' in model:
#        b_named['p'] = b[next_idx:next_idx + 3]
#        b_se_named['p'] = b_se[next_se_idx:next_se_idx + 3]
#        next_idx += 3
#        next_se_idx += 3
#    if 's' in model:
#        b_named['s'] = b[next_idx]
#        b_se_named['s'] = b_se[next_se_idx]
#        next_idx += 1
#        next_se_idx += 1
#    if ('v' in model) or ('d' in model): 
#        if not 'X' in model:
#            if 'd' in model:
#                b_named['d'] = b[next_idx:]
#                b_se_named['d'] = b_se[next_se_idx:]
#            elif 'v' in model:
#                b_named['v'] = b[next_idx:]
#                b_se_named['v'] = b_se[next_se_idx:]
#        else:
#            b_dir = b[next_idx:]
#            b_se_dir = b_se[next_se_idx:]
#            if 'd' in model:
#                b_named['d'] = b_dir.reshape(b_dir.shape[0]/3,3)
#                b_se_named['d'] = b_se_dir.reshape(b_se_dir.shape[0]/3,3)
#            elif 'v' in model:
#                b_named['v'] = b_dir.reshape(b_dir.shape[0]/3,3)
#                b_se_named['v'] = b_se_dir.reshape(b_se_dir.shape[0]/3,3)
#    return b_named, b_se_named, rsq

# def calc_vel(pos, time):
#     '''
#     Calculates velocity from position.

#     Parameters
#     ----------
#     pos : ndarray, shape (ntasks, nbins, ndims)
#       positional data from optotrak
#     time : ndarray, shape (ntasks, nbins)
#       bin edge times
      
#     Returns
#     -------
#     vels : ndarray, shape (n_tasks, n_bins, n_dims)
#       velocity data at same time points as positional data
#     '''
#     assert np.rank(pos) == 3
#     dp = np.diff(pos, axis=1)
#     dt = np.diff(time, axis=1)
#     warn("Not checked since divide by time implemented.")
#     return dp / dt[...,None]

# def glm_full_model_old(frs, pos, time, model='', family='poisson-log'):
#     '''
#     Uses a generalized linear model to fit coefficients to the relationship
#     between firing rates and kinematics according to a given model.
    
#     Parameters
#     ----------
#     frs : array_like
#       firing rates, shape (n_trials, n_bins)
#     pos : array_like
#       positional data, shape (n_trials, n_bins, n_dims)
#     time : array_lie
#       bin edge times, shape (n_trials, n_bins)
#     model : string
#       model

#     Returns
#     -------
#     b : dict
#       dictionary of coefficients of regression
#     b_se : ndarray
#       standard errors of regression coefficients
#       might need to be divided up into dictionary too
#     '''
#     frs = np.asarray(frs)
#     assert(np.rank(frs) == 2)
#     pos = np.asarray(pos)
#     assert(np.rank(pos) == 3)
#     time = np.asarray(time)
#     assert(np.rank(time) == 2)    
#     assert(type(model) == str)
#     assert(not np.any(np.isnan(frs)))
#     assert(family in ['binomial-log',
#                       'poisson-ident', 'poisson-log',
#                       'gaussian-ident', 'gaussian-log'])
                      
#     n_trials, n_bins = frs.shape
#     assert(pos.shape == (n_trials, n_bins + 1, 3))
#     #n_bins -= 1 # because of -1 for velocity calculation (offset = 1 wrt frs)
#     n_dims = pos.shape[-1]
#     n_reg = n_trials * n_bins
#     #print "n_reg %d" % (n_reg)
#     reg_frs = frs.reshape(n_reg)
#     vels = None
#     reg_vars = None

#     # add the time-invariant variables
#     if 'k' in model:
#         add_const = True
#     else:
#         add_const = False
#     if 'p' in model:
#         pos_flat = edge2cen(pos.reshape(n_reg, 3), axis=1)
#         # drop a bin to match v
#         reg_vars = add_reg_vars_1d(reg_vars, pos_flat)
#     if 's' in model:
#         if vels == None:
#             vels = calc_vel(pos, time)
#         spds = np.sqrt(np.sum(vels**2, axis=2))
#         spds_flat = spds.reshape(n_reg)
#         reg_vars = add_reg_vars_1d(reg_vars, spds_flat)

#     # do the potentially time-variant variables
#     if ('d' in model or 'v' in model):
#         if (vels == None):
#             vels = calc_vel(pos, time)
#         if 'd' in model:
#             speeds = np.sqrt(np.sum(vels**2, axis=2))
#             var = vels / speeds[...,None]
#         elif 'v' in model:
#             var = vels
#         if not 'X' in model: # time-static variable
#             var_flat = np.reshape(var, (n_reg, 3))
#         else: # time-dynamic variable
#             mask = np.eye(n_bins)
#             var_flat = np.reshape(mask[None,...,None] * var[...,None,:],
#                                   (n_reg, n_bins * n_dims))
#         reg_vars = add_reg_vars_1d(reg_vars, var_flat)
#     if add_const:
#         reg_vars = sm.tools.add_constant(reg_vars)

#     loglink = sm.families.links.Log()
#     if family == 'poisson-ident':
#         fam = sm.families.Poisson()
#     elif family == 'poisson-log':
#         fam = sm.families.Poisson(loglink)
#     elif family == 'gaussian-ident':
#         fam = sm.families.Gaussian()
#     elif family == 'gaussian-log':
#         fam = sm.families.Gaussian(loglink)
#     elif family == 'binomial-log':
#         fam = sm.families.Binomial(loglink)
#     else:
#         fam = None       
#     glmodel = sm.GLM(reg_frs, reg_vars, family=fam)
#     results = glmodel.fit()
#     b = results.params
#     bse = results.bse

#     # recover coefficients from regression product
#     b_nom = {}
#     bse_nom = {}
#     idx = 0
#     last_idx = None
#     if 'k' in model:
#         b_nom['k'] = b[-1]
#         last_idx = -1
#     if 'p' in model:
#         b_nom['p'] = b[idx:idx + 3]
#         bse_nom['p'] = bse[idx:idx + 3]
#         idx += 3
#     if 's' in model:
#         b_nom['s'] = b[idx]
#         bse_nom['s'] = bse[idx]
#         idx += 1
#     if ('v' in model) or ('d' in model):
#         if 'd' in model:
#                 dest = 'd'
#         elif 'v' in model:
#                 dest = 'v'
#         if not 'X' in model:
#             b_nom[dest] = b[idx:last_idx]
#             bse_nom[dest] = bse[idx:last_idx]
#         else:
#             b_dir = b[idx:last_idx]
#             bse_dir = bse[idx:last_idx]
#             b_nom[dest] = b_dir.reshape(b_dir.shape[0]/3,3)
#             bse_nom[dest] = bse_dir.reshape(bse_dir.shape[0]/3,3)
#     return b_nom, bse_nom

# def regress_all_bins(frs, pos, model='kvsp', with_err=True):
#     '''Calculate preferred directions from average values across movements.

#     Parameters
#     ----------
#     frs : array_like, shape (n_trials, n_bins)
#       firing rates (PSTHs)
#     pos : array_like, shape (n_trials, n_bins, 3) # maybe not +1??
#       positional data in same bins as PSTHs
#     model : string encoding which parameters to regress against
#       basically same code as used in `tuning_change.firing_models`
#       k = constant term
#       v = velocity
#       d = direction
#       s = speed
#       p = position
#       need to consider different kinds of direction, as in previous version

#     Returns
#     -------
#     b_named : dict
#       regression coefficients
#     [b_se : ndarray
#       std errors of regression co-efficients
#     r : ndarray
#       R-squared value of fit]
#     '''
#     assert np.rank(frs) == 2
#     assert np.rank(pos) == 3
#     n_trials, n_bins = frs.shape

#     assert pos.shape == (n_trials, n_bins, 3)
#     assert pos.shape[2] == 3
    
#     reg_frs = frs[:,1:].reshape(n_trials * (n_bins - 1))
#     vels = None
#     reg_vars = None
#     if 'k' in model:
#         add_const = True
#     else:
#         add_const = False
#     if 'd' in model:
#         vels = calc_vel(pos)
#         speeds = np.sqrt(np.sum(vels**2, axis=2))
#         dirs = vels / speeds[...,None]
#         dirs_flat = dirs.reshape(n_trials * (n_bins - 1), 3)
#         reg_vars = add_reg_vars_1d(reg_vars, dirs_flat)
#     elif 'v' in model:
#         if vels == None:
#             vels = calc_vel(pos)
#         vels_flat = vels.reshape(n_trials * (n_bins - 1), 3)
#         reg_vars = add_reg_vars_1d(reg_vars, vels_flat)
#     if 'p' in model:
#         pos_flat = pos[:,1:,:].reshape(n_trials * (n_bins - 1), 3)
#         reg_vars = add_reg_vars_1d(reg_vars, pos_flat)
#     if 's' in model:
#         if vels == None:
#             vels = calc_vel(pos)
#         spds = np.sqrt(np.sum(vels**2, axis=2))
#         spds_flat = spds.reshape(n_trials * (n_bins - 1))
#         reg_vars = add_reg_vars_1d(reg_vars, spds_flat)

#     # end game
#     next_idx = 0
#     b_named = {}
#     if not with_err:
#         b = multiregress(reg_vars, reg_frs,
#                          with_err=False, add_const=add_const)
#         if 'k' in model:
#             b_named['k'] = b[next_idx]
#             next_idx += 1
#         if 'd' in model:
#             b_named['d'] = b[next_idx:next_idx + 3]
#             next_idx += 3
#         elif 'v' in model:
#             b_named['v'] = b[next_idx:next_idx + 3]
#             next_idx += 3
#         if 'p' in model:
#             b_named['p'] = b[next_idx:next_idx + 3]
#             next_idx += 3
#         if 's' in model:
#             b_named['s'] = b[next_idx]
#         return b_named
#     else:
#         b, b_se = multiregress(reg_vars, reg_frs,
#                                with_err=True, add_const=add_const)
#         r = rsq_from_b(reg_vars, reg_frs, b, add_const=add_const)
#         if 'k' in model:
#             b_named['k'] = b[next_idx]
#             next_idx += 1
#         if 'd' in model:
#             b_named['d'] = b[next_idx:next_idx + 3]
#             next_idx += 3
#         elif 'v' in model:
#             b_named['v'] = b[next_idx:next_idx + 3]
#             next_idx += 3
#         if 'p' in model:
#             b_named['p'] = b[next_idx:next_idx + 3]
#             next_idx += 3
#         if 's' in model:
#             b_named['s'] = b[next_idx]        
#         return b_named, b_se, r

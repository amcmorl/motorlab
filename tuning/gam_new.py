import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from amcmorl_py_tools.proc_tools import edge2cen
from warnings import warn
#from amcmorl_py_tools.vecgeom import unitvec
from scipy import stats
from sklearn.cross_validation import KFold

import motorlab.kinematics as kin

ro.r('require(mgcv)')
rdir = '/home/amcmorl/lib/python/motorlab/tuning/r'
ro.r('source("%s/fit_gam.R")' % rdir)
rgam = ro.r('fit_gam')

sbt = "$_t$"
modellist = ["kd","kdp","kds","kdps","kv","kvp","kvs","kvps"]
pretty_modellist = modellist + \
    ["kd"+sbt,"kd"+sbt+"p","kd"+sbt+"s", "kd"+sbt+"ps",
     "kv"+sbt,"kv"+sbt+"p","kv"+sbt+"s","kv"+sbt+"ps"]
modellist += [x + 'X' for x in modellist]
modellist += ['null']
max_ncoef = 35

'''
Notes
-----
The R method doesn't differentiate between trials, i.e. splines are fit to all
trials together. This should be okay because the variables we are fitting, PD
and PV, are theoretically trial-independent, i.e. curves are the same across all
trials regardless of the direction of the individual trial. I'm not sure if this
would be different for fitting position, but would need to be considered.

To Do
-----
should really save number of samples used in each CV
'''

def _format_for_gam(count, time, pos):
    '''
    Format data for gam_predict_cv, i.e. an (n, 12) array
    
    Parameters
    ----------
    count : ndarray
      spike counts, shape (ntrial, nbin)
    time : ndarray
      bin_edges, shape (ntrial, nbin + 1)
    pos : ndarray
      positions at `bin_edges`, shape (ntrial, nbin + 1, 3)
    
    Returns
    -------
    formatted_data : ndarray
      shape (ntrial * nbin, 12)
      dimension 1 is [count, t, dx, dy, dz, px, py, pz, vx, vy, vz, sp]
    '''        
    assert np.rank(count) == np.rank(time) == (np.rank(pos) - 1)
    assert count.shape[0] == time.shape[0] == pos.shape[0]
    assert (count.shape[1] + 1) == time.shape[1] == pos.shape[1]

    y  = count.flatten()[:,None]
    npt = y.size
    tax, spax = 1, -1
    
    # don't use real time as won't compare equivalent portions of trials
    # t  = edge2cen(time, axis=tax) # (ntrial, nbin)
    # subtract offset to get all relative times
    #t = (t - t[:,:1]).flatten()[:,None]
    
    # instead use bin numbers
    ntrial, nbin = count.shape
    t = np.tile(np.arange(nbin, dtype=float)[None], (ntrial, 1))
    t = t.flatten()[:,None]
    
    d  = kin.get_dir(pos, tax=tax, spax=spax).reshape(npt, 3)
    p  = edge2cen(pos, axis=tax).reshape(npt, 3)
    v  = kin.get_vel(pos, time, tax=tax, spax=spax).reshape(npt, 3)
    sp = kin.get_speed(pos, time, tax=tax, spax=spax).flatten()[:,None]
    return np.concatenate([y,t,d,p,v,sp], axis=1)

class GAMOneModel(object):
    '''
    Parameters
    ----------
    rdata : RVector
        result from fit_gam.R's fit_gam
    train_time : ndarray
        time values of training data
        shape (ntrain_trials * nbin)
    test_shape : tuple
        shape of test data
    
    Notes
    -----
    from rdata:
        data[0] = coef, shape (ncoef)
        data[1] = coef_names, shape (ncoef)
        data[2] = smooths, shape (ntrain_pt, 3)
        data[3] = pred, shape (ntest_pt)
    '''    
    def __init__(self, rdata, train_time, test_shape):
        # primary data
        self.coef = np.array(rdata[0])
        # max 35, for nbin = 10
        self.coef_names = np.array(rdata[1])
        if (np.array(rdata[2]).size == 1):
            self.smooth = None
        else:
            sm = np.array(rdata[2])
            tm = train_time
            self.smooth = np.concatenate([tm[:,None], sm], axis=1)
            
        # ntrial * nbin * 3
        self.pred = np.array(rdata[3]).reshape(test_shape)
        # ntest, nbin
        self.logl = np.array(rdata[4])
        self.logl_df = np.array(rdata[5], dtype=int)

class GAMUnit(object):
    '''
    Attributes
    ----------
    actual : ndarray
      shape (ntask, nrep, nbin) actual counts in order of original data
    predicted : ndarray
      shape (ntask, nrep, nbin) predicted counts, same order as `actual`
    coefs : ndarray
      shape (nmodel, ncv, max_ncoef) coefficients fitted by GAM
      Okay because doesn't rely on trial number per CV
    coef_names : ndarray
      shape (nmodel, max_ncoef) names of coefficients fitted by GAM
      Okay because doesn't rely on trial number per CV,
      and is consistent across CVs
    smooth : ndarray
      shape (npoint, 4) smoothed PD or PV values from GAM fit
      smooth[0] is time point, smooth[1:] is x,y,z value
    perm : ndarray
      shape (ntrial) order of permuted trials presented to GAM
      permutation is required to get even sampling when dividing
      into test and training data
    metadata : sequence
      shape (dsname, unit, lag, align) metadata about this dataset

    Methods
    -------
    __init__(actual, permutation, ncv, dsname, unit, lag, align)
        instantiate a GAMUnit with the data from R calculation
    __setitem__(idx, data)
        set, across several attributes, values at index `idx`
    save(file_name)
        save object into file `file_name`
    '''
    def __init__(self, actual, permutation, ncv,
                 dsname, unit, lag, align):
        # metadata must be constant
        self.dsname = dsname
        self.unit = unit
        self.lag = lag
        self.align = align

        # preknown data        
        self.actual = actual
        self.perm = permutation
        
        # acquired data
        nmodel = len(modellist)        
        self.coef         = np.zeros((nmodel, ncv, max_ncoef)) + np.nan
        self.coef_names   = np.zeros((nmodel, max_ncoef), dtype='|S11')
        nsmooth_model     = nmodel / 2
        self.smooth       = np.zeros((nsmooth_model, actual.size, 4))  + np.nan
        # keep track of next place to insert smooth, per model
        self.smooth_place = np.zeros((nsmooth_model))
        
        # will have `nmodel` points per point in original data
        self.pred         = np.zeros([nmodel,] + list(actual.shape)) + np.nan
        self.logl         = np.zeros([nmodel, ncv]) + np.nan
        self.logl_df      = np.zeros([nmodel, ncv], dtype=int)

    def save(self, file_name):
        '''
        Serialize self to a numpy npz file

        Parameters
        ----------
        file_name : string
          name of file to create
        '''
        data = {'dsname'     : self.dsname,
                'unit'       : self.unit,
                'lag'        : self.lag,
                'align'      : self.align,
                'coef'       : self.coef,
                'coef_names' : self.coef_names,
                'smooth'     : self.smooth,
                'actual'     : self.actual,
                'pred'       : self.pred,
                'perm'       : self.perm,
                'ncv'        : self.coef.shape[1],
                'logl'       : self.logl,
                'logl_df'    : self.logl_df}
        np.savez(file_name, **data)

    def __setitem__(self, idx, data):
        '''
        Parameters
        ----------
        idx : tuple
          index of (imodel, icv)
        data : tuple
          gam_out, test
        
        where
        
        gam_out : GAMOneModel
          result of a single call to fit_gam.R's fit_gam
          has
            coef - insert into correct CV slot
            coef_names - check same as existing, per model
            smooth - concatenate along pts axis (0)
            pred - use permutation to put values in appropriate place
              relative to actual
            logl - loglikelihood of prediction

        Notes
        -----        
        overrides behaviour of bla[idx] = data 
        '''
        gam_out, test = data
        model, cvidx = idx
        if not type(gam_out) == GAMOneModel:
            raise ValueError("`data[1]` must be an instance of GAMOneModel")
        # prediction - same shape for all models
        # coef and coef names - different for each model
        # smooth - different for each model
        #   all smooth values lie on same smoothed curve, at different
        #   time points depending on time of trial (bins)
        if not idx[0] in modellist:
            raise ValueError("`model` not known.")
        midx = modellist.index(model)
        
        self.coef[midx, cvidx,:len(gam_out.coef)] = gam_out.coef
        
        if self.coef_names[midx,0] == '':
            self.coef_names[midx, :len(gam_out.coef_names)] = \
                gam_out.coef_names
        else:
            if not np.all(self.coef_names[midx, :len(gam_out.coef_names)] \
                              == gam_out.coef_names):
                raise ValueError("coef_names are not as expected")
                
        # assume kdX is the first smooth model, 
        # and all unsmooth models precede it
        nunsmooth_models = modellist.index('kdX')
        if 'X' in model:
            smidx = midx - nunsmooth_models
            start = self.smooth_place[smidx]
            stop = start + gam_out.smooth.shape[0]
            self.smooth[smidx, start:stop] = gam_out.smooth
                       
        # self.pred has nmodel, ntrial, nbin
        # need to make a copy of self.pred[midx] permuted by `self.perm`,
        # ... assign into that at `test` locations
        # ... and then assign
        permuted = self.pred[midx, self.perm]
        permuted[test] = gam_out.pred
        unpermuted = self.pred[midx]
        unpermuted[self.perm] = permuted
        self.pred[midx] = unpermuted
        self.logl[midx, cvidx] = gam_out.logl
        self.logl_df[midx, cvidx] = gam_out.logl_df
                        
def load_gam_unit(file_name):
    gf = np.load(file_name)   
    warn('using default ncv=10')     
    gam_unit = GAMUnit(gf['actual'],
                       gf['perm'],
                       10,
                       gf['dsname'],
                       gf['unit'],
                       gf['lag'],
                       gf['align'])
    gam_unit.pred    = gf['pred']
    gam_unit.smooth  = gf['smooth']
    gam_unit.coef    = gf['coef']
    gam_unit.coef_names = gf['coef_names']
    gam_unit.logl    = gf['logl']
    gam_unit.logl_df = gf['logl_df']
    return gam_unit
        
def fold_repeats(out, bnd):
    '''
    Fold flattened predictions from GAM back into tasks and repeats.    
    
    Works inplace.
    '''
    nmodel, ntrial, nbin = out.pred.shape
    notnans = bnd.get_notnans()
    pred_with_nans = np.zeros([nmodel,len(notnans), nbin]) + np.nan
    pred_with_nans[:,notnans] = out.pred
    pred_folded = np.reshape(pred_with_nans, [nmodel,] + list(bnd.count.shape))
    
    # and do for actual to confirm all is okay
    #actual_with_nans = np.zeros([len(notnans), nbin]) + np.nan
    #actual_with_nans[notnans] = out.actual
    #actual_folded = np.reshape(actual_with_nans, bnd.count.shape)
    #np.testing.assert_array_equal(actual_folded.shape, bnd.count.shape)
    #np.testing.assert_array_almost_equal(actual_folded.flatten()[notnans],
    #                               bnd.count.flatten()[notnans])
    out.pred = pred_folded
    out.actual = bnd.count
    
def gam_predict_cv(bnd, models, metadata, nfold=10, family='poisson',
    verbosity=0):
    '''
    Parameters
    ----------
    bnd : BinnedData
      class containing counts and pos to fit
    models : list of string
      models to do gam
    metadata : list
      [dsname, unit, lag, align]
    nfold : int
      number of CV folds to perform
    family : string
      'poisson' or 'gaussian', noise model to assume when fitting
    verbosity : int
      level of reporting to do (0=None, 1=CV, 2=CV,model)
    Notes
    -----
    Problem with KFold cross-validation is that last one doesn't necessarily
    have same number of trials as previous. Two solutions: 1) skip last cross
    validation; 2) find a factor (19 in this case) which divides evenly.
    Starting with solution #1. Third solution might be to slot data back into
    its place re: task direction, since every trial is then estimated using 
    cross-validation, and then take average of all trials that way.
    '''
    #pp = PosPlotter()    
    
    # check models are valid
    for model in models:
        if not model in modellist:
            raise ValueError("unknown model: %s" % (model))
    
    count, pos, time = bnd.get_for_glm(0)
 
    # all trials will get predicted eventually    
    
    # Since KFold samples systematically, I need to mix up data in a 
    # reproducible way first.
    # Fancy indexing always creates a copy, so will need a way to get back
    # again from permuted form.
    # Can be done using the following:
        # permuted = data[permutation]
        # ... do stuff to permuted
        # ... and put it back again
        # data[permutation] = permuted    
    ntrial, nbin = count.shape
    perm = np.random.permutation(range(ntrial))

    gam_unit = GAMUnit(count, perm, nfold, *metadata)
    
    # select subsets of trials for training and testing
    folds = KFold(ntrial, nfold)
    for icv, (train, test) in enumerate(folds):
        #if icv != 0:
        #    print "only doing first CV!"
        #    continue
        if verbosity > 0:
            print "Processing cross-validation #%d" % (icv)
        # format training data to:
        # y, t, dx, dy, dz, px, py, pz, vx, vy, vz, sp
        # and all the same length
        
        #_ = count[perm][train].shape
        #print "in cv, count train: %d x %d " % (_[0], _[1])
        #_ = count[perm][test].shape
        #print "in cv, count test: %d x %d " % (_[0], _[1])
        if verbosity > 0:
            print "mean count: %0.3f" % (np.mean(count[perm][train]))
        
        data_train = _format_for_gam(count[perm][train], 
                                     time[perm][train],
                                     pos[perm][train])
        #print np.allclose(data_train[:,1], 0) # should be all zeros
        
        data_test = _format_for_gam(count[perm][test],
                                    time[perm][test],
                                    pos[perm][test])
        #np.savez('data-for-gam.npz', data_train=data_train,
        #         data_test=data_test)
        # have to format _after_ selecting training/testing data so that
        # trials are selected together (since formatting flattens time pts)
        #pp.plot(pos[perm][test])
       
        # loop over models
        for model in models:
            if verbosity > 1:
                print "Processing %s" % (model)
            # do gam, get prediction
            rout = rgam(data_train, data_test, model, True, family)
            gam_one = GAMOneModel(rout, data_train[:,1], count[test].shape)
            
            # put in container            
            gam_unit[model, icv] = gam_one, test
    
    # now fold gam_unit.pred and gam_unit.actual back into tasks/repeats
    fold_repeats(gam_unit, bnd)
    return gam_unit
    
def near(arr, num, err=1e-8):
    return np.abs(arr - num) < err    
    
def get_average_smooth(smooth):
    '''
    Parameters
    ----------
    smooth : array_like
      shape (npt, 4), a smooth trajectory from one model
    '''
    if np.rank(smooth) != 2:
        raise ValueError('smooth needs to have 2 dimensions exactly')
    # get unique time points - since are bins, should just be the ordinals
    tall = smooth[:,0]
    t = np.unique(tall)
    t = t[~np.isnan(t)]

    xyzall = smooth[:,1:]
   
    avg_shape = list(t.shape) + [3,]
    xyz = np.zeros(avg_shape)
    std = np.zeros(avg_shape)
    for i, pt in enumerate(t):
        #pts = near(t, pt)
        pts = np.isclose(t, pt)
        xyz[i] = np.mean(xyzall[pts], axis=0)
        std[i] = np.std(xyzall[pts], axis=0)
    return t, xyz, std

def get_model_from_params(p):
    '''
    Construct model string (e.g. kdpX) from gam returned params.
    '''
    model = ''
    dynamic = False
    if '(Intercept)' in p:
        model += 'k'
    if 'dx' in p:
        model += 'd'
    if 'vx' in p:
        model += 'v'
    if 's(t):dx.1' in p:
        model += 'd'
        dynamic = True
    if 's(T):vx.1' in p:
        model += 'v'
        dynamic = True
    if 'px' in p:
        model += 'p'
    if 'sp' in p:
        model += 's'
    if dynamic:
        model += 'X'
    return model

def unpack_coefficients(p, pname):
    '''
    From one model's worth of coefs `p` and coef_names `pnames` components of
    a GamUnit, sort the coefficients into a dictionary equivalent to that from
    `tuning.fit`
    '''
    dims = ['x', 'y', 'z']
    ncv, nmax_coef = p.shape
    # `p` has shape ncv, nmax_coef
    # `pname` has shape nmax_coef
    # if coef_names has P, calculate angle between P and D (or V)
    bdict = {}
    
    if '(Intercept)' in pname:
        bdict['k'] = p[:,np.flatnonzero(pname == '(Intercept)')].squeeze()

    # identify 'd' or 'v', and 'X' or not 'X'
    if ('dx' in pname):
        d, dynamic = 'd', False
    elif ('s(t):dx.1' in pname):
        d, dynamic = 'd', True
    elif ('vx' in pname):
        d, dynamic = 'v', False
    elif ('s(t):vx.1' in pname):
        d, dynamic = 'v', True
    else:
        d, dynamic = 'none', False
    #    raise ValueError('Oops - all models should have "d"'
    #                     ' or "v" in them, instead have %s.' % (pname))
    if (d != 'none'):
        if not dynamic:
            dx = p[:,np.flatnonzero(pname == '%sx' % (d))].squeeze()
            dy = p[:,np.flatnonzero(pname == '%sy' % (d))].squeeze()
            dz = p[:,np.flatnonzero(pname == '%sz' % (d))].squeeze()
            bdict[d] = np.array([dx, dy, dz])
        else:
            # identify number of bins
            pattern = 's(t):%sx' % (d)
            nbin = np.sum([pattern in x for x in pname])
            
            darr = np.zeros((3, ncv, nbin))
            bdict[d] = darr
            for i, dim in enumerate(dims):
                first, last = ['s(t):%s%s.%s' % (d, dim, j) \
                                   for (j) in [1, nbin]]
                fidx = np.flatnonzero(pname == first)[0]
                lidx = np.flatnonzero(pname == last)[0]
                darr[i] = p[:, fidx:lidx + 1]
                
                #darr_av = np.mean(darr, axis=1)
                #darr_ = unitvec(darr_av, axis=0)
        
    if 'px' in pname:
        # get preferred position
        px = p[:,np.flatnonzero(pname == 'px')].squeeze()
        py = p[:,np.flatnonzero(pname == 'py')].squeeze()
        pz = p[:,np.flatnonzero(pname == 'pz')].squeeze()
        
        bdict['p'] = np.array([px,py,pz]) # has shape ndim, ncv
        
    if 'sp' in pname:
        bdict['s'] = p[:, np.flatnonzero(pname == 'sp')].squeeze()

    return bdict

# ----------------------------------------------------------------------------
# comparison of predicted and actual counts / values
# ----------------------------------------------------------------------------

def compress_late_dims(arr):
    '''
    Flatten all but first dim of `arr`, i.e. shape (3,2,4) becomes (3,8).
    '''
    n = arr.shape[0]
    return np.reshape(arr, [n, np.prod(arr.shape[1:])])

def get_mean_sem_of_samples(arr):
    '''
    Compress all dims > 0 to a single mean and SEM.
    '''
    arr_ = compress_late_dims(arr)
    mean = stats.nanmean(arr_, axis=1)
    N = np.sum(~np.isnan(arr_), axis=1)
    sem = stats.nanstd(arr_, axis=1) / np.sqrt(N)
    return mean, sem, N

def get_sum_of_samples(arr):
    '''
    Compress all dims > 0 to a single sum.
    '''
    arr_ = compress_late_dims(arr)
    return np.nansum(arr_, axis=1)

def calc_sample_loglik(gam_unit, family='poisson'):
    '''
    Calculate log likelihood loss function for Poisson or Gaussian
    distributed data.

    Log likelihood is defined as:

    .. math:: L(Y, \theta(X)) = -2 \cdot \log \mathtext{Pr}_{\theta(X)}(Y)
    
    where :math:`\theta(X)` is the prediction and :math:`Y` is the actual data.
    From Friedman, Tibshirani, and Hastie, 2nd ed, 5th print, eq. 7.8.
    This is the probability of seeing the data, given the prediction.
    
    For Poisson, :math:`\mathtext{Pr}_{\theta(X)}(Y)` is given by 
    :math:`pmf(Y,f(X))` and for gaussian, by :math:`pdf(Y, f(X))`
    '''
    assert family in ['poisson', 'gaussian']
    
    if family == 'poisson':
        # only needs mean shape parameter
        Pr = -2 * stats.poisson.logpmf(gam_unit.actual, gam_unit.pred)
    else:
        # normal needs means and variance
        # shape handling here calculates average across repeats,
        # but keeps that dimension for broadcasting purposes
        pred_std = stats.nanstd(gam_unit.pred, axis=2)[:,:,None]
        Pr = -2 * stats.norm.logpdf(gam_unit.actual, gam_unit.pred, pred_std)
    
    # Pr now has shape (nmod, ntask, nrep, nunit, nbin)
    # get one number per model...
    return get_mean_sem_of_samples(Pr)
    
def calc_loglik(gam_unit, family='poisson'):
    '''
    Calculate log likelihood for Poisson or Gaussian distributed data.
    '''
    assert family in ['poisson', 'gaussian']
    
    if family == 'poisson':
        # only needs mean shape parameter
        Pr = stats.poisson.logpmf(gam_unit.actual, gam_unit.pred)
    else:
        # normal needs means and variance
        # shape handling here calculates average across repeats,
        # but keeps that dimension for broadcasting purposes
        pred_std = stats.nanstd(gam_unit.pred, axis=2)[:,:,None]
        Pr = stats.norm.logpdf(gam_unit.actual, gam_unit.pred, pred_std)
        
    # Pr now has shape (nmod, ntask, nrep, nunit, nbin)
    # get one number per model...
    return get_sum_of_samples(Pr)
    
def calc_kendall_tau(gam_unit, average=False):
    '''
    Calculate Kendall tau value for predicted values. This tau scales between
    -1 (prefect negative correlation) and 1 (perfect correlation). 
    
    gam_unit : GamUnit
      has `actual` and `pred` attributes
    average : bool
      average across repeats before calculating tau
    '''
    assert(type(average) == bool)
    
    if not average:
        act_flat = gam_unit.actual.flatten()
    else:
        act_flat = stats.nanmean(gam_unit.actual, axis=1).flatten()
    nans = np.isnan(act_flat)
    act_flat = act_flat[~nans]

    tau = np.zeros((gam_unit.pred.shape[0])) + np.nan
    P = np.zeros_like(tau) + np.nan
    for i, pred in enumerate(gam_unit.pred):
        if not average:
            pred_flat = pred.flatten()[~nans]
        else:
            pred_flat = stats.nanmean(pred, axis=1).flatten()
        tau[i], P[i] = stats.kendalltau(act_flat, pred_flat)
    return tau, P

def calc_mse(gam_unit):
    '''
    Calculate the Mean Squared Error (MSE) of a prediction.
    
    Is the mean across samples of the squared difference between actual and
    predicted values.
    
    .. math:: \text{MSE} = <(Y_{\mathrm{act} - Y_{pred})^2>_\mathrm{n}
    '''
    actual = gam_unit.actual
    pred = gam_unit.pred
    se = (actual[None] - pred)**2
    nmodel = se.shape[0]
    se = np.reshape(se, [nmodel, np.prod(se.shape[1:])])
    mse = stats.nanmean(se, axis=1)
    return mse

def calc_nagelkerkes_r2(gam_unit):
    '''
    Calculate Nagelkerke's R^2 [1].
    
    Nagelkerke NJD (1991). A Note on a General Definition of 
    the Coefficient of Determination. Biometrika 78, 691-692.
    
    Notes
    -----
    
    Nagelkerke's R2 is given by:
    
    .. math:: R^2 = \frac{1 - \frac{L(0)}{L(\hat\theta)^{2/n}}}{1-(L(0))^{2/n}}
    '''
    # check last model is intercept-only
    assert(gam_unit.coef_names[-1][0] == '(Intercept)')
    assert(np.all(gam_unit.coef_names[-1][1:] == ''))
    
    logl0 = gam_unit.logl[-1] # log likelihood for null model, shape 10
    loglm = gam_unit.logl     # log likelihood for other models, shape (17,10)
    
    # for some reason, actual # non-nans > pred # non-nans
    n  = (~np.isnan(gam_unit.pred[0])).sum().astype(float)

    # do most of calculations in log space to avoid numeric over/underflows
    numerator   = 1 - np.exp(2/n * (logl0 - loglm))
    denominator = 1 - np.exp(2/n * logl0)
    R2s = numerator / denominator
    
    # take mean across cross-validations
    # should really use a weighted mean, 
    # taking into account size of each CV
    return np.mean(R2s, axis=1)
    
    

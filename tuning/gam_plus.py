import os
from warnings import warn

import numpy as np
from scipy import stats
from sklearn.cross_validation import KFold
import h5py
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from amcmorl_py_tools.proc_tools import edge2cen
import motorlab.kinematics as kin

ro.r('require(mgcv)')
rdir = '/home/amcmorl/lib/python/motorlab/tuning/r'
ro.r('source("%s/fit_gam_plus.R")' % rdir)
rgam = ro.r('fit_gam_plus')

sbt = "$_t$"
modellist = ["kd","kdp","kds","kdps","kv","kvp","kvs","kvps"]
pretty_modellist = modellist + \
    ["kd"+sbt,"kd"+sbt+"p","kd"+sbt+"s", "kd"+sbt+"ps",
     "kv"+sbt,"kv"+sbt+"p","kv"+sbt+"s","kv"+sbt+"ps"]
modellist += [x + 'X' for x in modellist] + ['kqX']
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

    # q is a second direction set-of-columns for deviance calculation
    q  = kin.get_dir(pos, tax=tax, spax=spax).reshape(npt, 3)
    return np.concatenate([y,t,d,p,v,sp,q], axis=1)

def _format_for_gam2(count, time, pos):
    assert np.rank(count) == np.rank(time) == (np.rank(pos) - 1)
    assert count.shape[0] == time.shape[0] == pos.shape[0]
    assert (count.shape[1] + 1) == time.shape[1] == pos.shape[1]

    y = count.flatten()[:,None]
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

    # also create trial numbers for pulling out appropriate later on
    # these don't correspond to original trial numbers because original trials
    # have been permuted before getting here
    tr = np.tile(np.arange(ntrial), (nbin, 1)).T.flatten()
    
    d  = kin.get_dir(pos, tax=tax, spax=spax).reshape(npt, 3)
    p  = edge2cen(pos, axis=tax).reshape(npt, 3)
    v  = kin.get_vel(pos, time, tax=tax, spax=spax).reshape(npt, 3)
    sp = kin.get_speed(pos, time, tax=tax, spax=spax).flatten()[:,None]

    # q is a second direction set-of-columns for deviance calculation
    q  = kin.get_dir(pos, tax=tax, spax=spax).reshape(npt, 3)
    return np.concatenate([y,t,d,p,v,sp,q], axis=1), tr

class GAMFoldModel(object):
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
    def __init__(self, rdata, train_time, test_shape, test_idx):
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
        self.test_idx = test_idx

class GAMFitOneModel(object):
    '''
    Container for all the cross-valiations of the fit data
    (coefficients and predictions) to one dataset with one model.
    '''
    def __init__(self, parent, model):
        '''
        Parameters
        ----------
        model : string
          code for this model
        parent : GAMFitManyModels
          collection of fits for this dataset (may have only one fit, in fact)
        '''
        self.model  = model
        self.parent = parent
        ncv = self.parent.ncv

        nbin = parent.actual.shape[-1]
        nparam = _get_nparam(model, nbin)
        self.coef = np.empty([ncv, nparam]); self.coef.fill(np.nan)
        self.coef_names = np.zeros([nparam], dtype='|S11')

        if 'X' in model:
            self.smooth_place = 0
            self.smooth = np.empty([self.parent.actual.size, 4])
            self.smooth.fill(np.nan)
        
        self.pred = np.empty(self.parent.actual.shape)
        self.pred.fill(np.nan)

        self.logl = np.empty([ncv]); self.logl.fill(np.nan)
        self.logl_df = np.empty([ncv]); self.logl_df.fill(np.nan)

    @classmethod
    def from_file(cls, parent, model, f):
        me            = cls(parent, model)
        fcoef = f['coef']
        me.coef       = np.asarray(fcoef)
        me.coef_names = fcoef.attrs['names']
        if 'X' in model:
            me.smooth = np.asarray(f['smooth'])
        me.pred       = np.asarray(f['pred'])
        flogl = f['logl']
        me.logl       = np.asarray(flogl)
        me.logl_df    = flogl.attrs['df']
        return me

    @classmethod
    def from_npz_file(cls, parent, model, i, f):
        me            = cls(parent, model)
        coef = f['coef'][i]
        valid_coef_mask = ~np.all(np.isnan(coef), axis=0)
        me.coef       = np.array(coef[:,valid_coef_mask])
        coef_names = f['coef_names'][i]
        me.coef_names = np.array(coef_names[valid_coef_mask])
        if 'X' in model:
            nconst_models = 8
            me.smooth = np.asarray(f['smooth'][i - nconst_models])
        me.pred       = np.asarray(f['pred'][i])
        me.logl       = np.asarray(f['logl'][i])
        me.logl_df    = np.asarray(f['logl_df'][i])
        return me
        
    def _set_coef_names(self, names):
        n = len(names)
        if self.coef_names[0] == '':
            self.coef_names[:n] = names
        else:
            # check names match
            if not np.all(self.coef_names[:n] == names):
                raise ValueError("coef names are not as expected")

    def _set_smooth(self, smooth):
        start = self.smooth_place
        stop  = start + smooth.shape[0]
        self.smooth[start:stop] = smooth

    def _set_pred(self, predn, idx):
        # self.pred has shape (ntrial, nbin)
        # need to make a copy of `self.pred` permuted by `self.perm`,
        # ... assign into that at `idx` locations
        # ... and then assign that back into unpermuted using fancy indexing
        perm             = self.parent.perm
        permuted         = self.pred[perm]
        permuted[idx]    = predn
        unpermuted       = self.pred
        unpermuted[perm] = permuted
        self.pred        = unpermuted

    def _set_logl(self, cvidx, logl, df):
        self.logl[cvidx]    = logl
        self.logl_df[cvidx] = df

    def incorp(self, cvidx, data):
        '''
        cvidx : int
          index of this cross-validation
        data : GAMFoldModel
          coefs and prediction from one fold with one model
        '''
        self.coef[cvidx,:len(data.coef)] = data.coef
        self._set_coef_names(data.coef_names)
        if 'X' in self.model:
            self._set_smooth(data.smooth)
        self._set_pred(data.pred, data.test_idx)
        self._set_logl(cvidx, data.logl, data.logl_df)

    def fold_repeats(self, bnd):
        ntrial, nbin = self.pred.shape
        notnans = bnd.get_notnans()
        pred_with_nans = np.zeros([len(notnans), nbin]) + np.nan
        pred_with_nans[notnans] = self.pred
        pred_folded = np.reshape(pred_with_nans, list(bnd.count.shape))
        self.pred = pred_folded

    def save(self, f):
        '''
        f : H5py.File
          handle to file to store in
        '''
        g = f.create_group(self.model)
        coef = g.create_dataset('coef', data=self.coef)
        coef.attrs['names'] = self.coef_names
        if 'X' in self.model:
            g.create_dataset('smooth', data=self.smooth)
        g.create_dataset('pred',   data=self.pred)
        logl = g.create_dataset('logl', data=self.logl)
        logl.attrs['df'] = self.logl_df

class GAMFitManyModels(object):
    '''
    Container for the fits to one dataset by several different models.
    
    Attributes
    ----------
    actual : ndarray
      shape (ntask, nrep, nbin) actual counts in order of original data
    pred : dict of ndarray of float
      predicted counts
      keys are model names, values are equivalent to `actual`
    coefs : dict of ndarray of float
      coefficients fitted by GAM
      keys are model names, values are coefficients, shape (ncv, max_ncoef) 
      Okay because doesn't rely on trial number per CV
    coef_names : ndarray
      shape (nmodel, max_ncoef) names of coefficients fitted by GAM
      Okay because doesn't rely on trial number per CV,
      and is consistent across CVs
    smooth : ndarray
      **unchanged** shape (npoint, 4) smoothed PD or PV values from GAM fit
      smooth[0] is time point, smooth[1:] is x,y,z value
    perm : ndarray
      **unchanged** shape (ntrial) order of permuted trials presented to GAM
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
        self.unit   = unit
        self.lag    = lag
        self.align  = align
        
        # preknown data
        self.ncv    = ncv
        self.actual = actual
        self.perm   = permutation

        # acquired data
        # old way of doing it:
        # nmodel        = len(modellist)
        # * coef          = np.zeros((nmodel, ncv, max_ncoef)) + np.nan
        # * coef_names    = np.zeros((nmodel, max_ncoef), dtype='|S11')
        # nsmooth_model = nmodel / 2
        # * smooth        = np.zeros((nsmooth_model, actual.size, 4)) + np.nan
        # keep track of next place to insert smooth, per model
        # * smooth_place  = np.zeros((nsmooth_model))
        # will have `nmodel` points per point in original data
        # * pred          = np.zeros([nmodel,] + list(actual.shape)) + np.nan
        # * logl          = np.zeros([nmodel, ncv]) + np.nan
        # * logl_df       = np.zeros([nmodel, ncv], dtype=int)
        self.fits = dict()

    def get_fit_object(self, model):
        if not self.fits.has_key(model):
            self.fits[model] = GAMFitOneModel(self, model)
        return self.fits[model]
    
    def __setitem__(self, idx, data):
        '''Display
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
        model, cvidx = idx
        if not type(data) == GAMFoldModel:
            raise ValueError("`data` must be an instance of GAMOneModel")
        # prediction - same shape for all models
        # coef and coef names - different for each model
        # smooth - different for each model
        #   all smooth values lie on same smoothed curve, at different
        #   time points depending on time of trial (bins)
        if not model in modellist:
            raise ValueError("model %s not known." % (model))

        fit = self.get_fit_object(model)
        fit.incorp(cvidx, data)

    def fold_repeats(self, bnd):
        '''
        Fold flattened predictions from GAM back into tasks and repeats.    
        
        Works inplace.
        '''
        for fit in self.fits.values():
            fit.fold_repeats(bnd)
        
        # and do for actual to confirm all is okay
        #actual_with_nans = np.zeros([len(notnans), nbin]) + np.nan
        #actual_with_nans[notnans] = out.actual
        #actual_folded = np.reshape(actual_with_nans, bnd.count.shape)
        #np.testing.assert_array_equal(actual_folded.shape, bnd.count.shape)
        #np.testing.assert_array_almost_equal(actual_folded.flatten()[notnans],
        #                               bnd.count.flatten()[notnans])
        self.actual = bnd.count

    def save(self, file_name):
        '''
        Serialize self to a numpy npz file

        Parameters
        ----------
        file_name : string
          name of file to create
        '''
        # check file doesn't already exist
        if os.path.exists(file_name):
            raise ValueError('file %s already exists.' % (file_name))
    
        # create file (automatically closes at end of block)
        with h5py.File(file_name, 'w-') as f:
            acl = f.create_dataset('actual', data=self.actual)
            acl.attrs['dsname'] = self.dsname
            acl.attrs['unit']   = self.unit
            acl.attrs['lag']    = self.lag
            acl.attrs['align']  = self.align
            acl.attrs['ncv']    = self.ncv
            f.create_dataset('perm', data=self.perm)
            for model, fit in self.fits.iteritems():
                fit.save(f)

    @classmethod
    def from_file(cls, file_name):
        f = h5py.File(file_name, 'r')
        fact   = f['actual']
        dsname = fact.attrs['dsname']
        unit   = fact.attrs['unit']
        lag    = fact.attrs['lag']
        align  = fact.attrs['align']
        ncv    = fact.attrs['ncv']
        actual = np.asarray(fact)
        perm   = np.asarray(f['perm'])
        me = cls(actual, perm, ncv, dsname, unit, lag, align)

        for model in _get_models(f.keys()):
            fit = GAMFitOneModel.from_file(me, model, f[model])
            me.fits[fit.model] = fit
        return me

    @classmethod
    def from_npz_file(cls, file_name):
        f = np.load(file_name)
        actual = f['actual']
        dsname = f['dsname']
        unit   = f['unit']
        lag    = f['lag']
        align  = f['align']
        ncv    = f['coef'].shape[1]
        perm   = f['perm']
        me = cls(actual, perm, ncv, dsname, unit, lag, align)

        # start off assuming all models are present
        pred = f['pred']
        nmodel = pred.shape[0]
        if nmodel == 16:
            fmodellist = modellist[:-2]
        elif nmodel == 17:
            # assume the missing model is kqX
            fmodellist = modellist[:-2] + modellist[-1:]
        elif nmodel == 18:
            fmodellist = modellist
        else:
            raise ValueError("Don't know how to handle %d models" % (nmodel))

        # order is as per modellist
        for i, model in enumerate(fmodellist):
            fit = GAMFitOneModel.from_npz_file(me, model, i, f)
            me.fits[fit.model] = fit
        return me

    def __eq__(self, other):
        #print type(self), type(other)
        return _compare_dict_same(self.__dict__, other.__dict__)
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

def _compare_dict_same(a, b):
    if isinstance(b, type(a)):
        for k, v in a.iteritems():
            o = b[k]
            if not _compare_same(v, o, k):
                print "%s is not equal" % (k)
                return False
        return True
    return NotImplemented

def _compare_same(a, b, name=''):
    if name == 'parent':
        return True
    elif type(a) == np.ndarray:
        if np.isscalar(a):
            return np.isclose(a, b)
        elif a.dtype.kind == 'S':
            if a.size == 1:
                return a == b
            else:
                return np.all(a == b)
        elif not np.all(np.isclose(a, b, equal_nan=True)):
            return False
        return True
    elif type(a) == dict:
        return _compare_dict_same(a, b)
    elif (hasattr(a, '__dict__') and (name != 'parent')):
        return _compare_dict_same(a.__dict__, b.__dict__)
    else:
        return a == b
        
def _get_models(keys):
    for k in keys:
        if not k in ['actual', 'perm']:
            yield k

def load_gam_unit(file_name):
    gf = np.load(file_name)   
    warn('using default ncv=10')     
    gam_unit = GAMFitManyModels(gf['actual'],
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
    #     permuted = data[permutation]
    #     ... do stuff to permuted
    #     ... and put it back again
    #     data[permutation] = permuted    
    ntrial, nbin = count.shape
    np.random.seed(43)
    perm = np.random.permutation(range(ntrial))

    count_ = count[perm]
    time_  = time[perm]
    pos_   = pos[perm]

    # initialize data store
    gam_unit = GAMFitManyModels(count, perm, nfold, *metadata)
    
    # select subsets of trials for training and testing
    folds = KFold(ntrial, nfold)
    for icv, (train, test) in enumerate(folds):
        if verbosity > 0:
            print "Processing cross-validation #%d" % (icv)
        # format training data to:
        # y, t, dx, dy, dz, px, py, pz, vx, vy, vz, sp
        # and all the same length
        
        if verbosity > 0:
            print "mean count: %0.3f" % (np.mean(count_[train]))
        
        data_train = _format_for_gam(count_[train], 
                                     time_[train],
                                     pos_[train])
        
        data_test = _format_for_gam(count_[test],
                                    time_[test],
                                    pos_[test])
        # have to format _after_ selecting training/testing data so that
        # trials are selected together (since formatting flattens time pts)
       
        # loop over models
        for model in models:
            if verbosity > 1:
                print "Processing %s" % (model)
            # do gam, get prediction
            rout = rgam(data_train, data_test, model, True, family)
            gam_fold_model = GAMFoldModel(rout, data_train[:,1], \
                count[test].shape, test)
            
            # put in container         
            gam_unit[model, icv] = gam_fold_model
    
    # now fold gam_unit.pred and gam_unit.actual back into tasks/repeats
    gam_unit.fold_repeats(bnd)
    return gam_unit

def gam_predict_cv2(bnd, models, metadata, nfold=10, family='poisson',
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
    #     permuted = data[permutation]
    #     ... do stuff to permuted
    #     ... and put it back again
    #     data[permutation] = permuted    
    ntrial, nbin = count.shape
    np.random.seed(43)
    perm = np.random.permutation(range(ntrial))

    count_ = count[perm]
    time_  = time[perm]
    pos_   = pos[perm]
    perm_format, perm_trial = _format_for_gam2(count_, time_, pos_)

    # initialize data store
    gam_unit = GAMFitManyModels(count, perm, nfold, *metadata)
    
    # select subsets of trials for training and testing
    folds = KFold(ntrial, nfold)
    for icv, (train, test) in enumerate(folds):
        if verbosity > 0:
            print "Processing cross-validation #%d" % (icv)
        # format training data to:
        # y, t, dx, dy, dz, px, py, pz, vx, vy, vz, sp
        # and all the same length
        
        if verbosity > 0:
            print "mean count: %0.3f" % (np.mean(count_[train]))

        train_idx = np.in1d(perm_trial, train)
        test_idx  = np.in1d(perm_trial, test)
        data_train = perm_format[train_idx]
        data_test  = perm_format[test_idx]
        
        # have to format _after_ selecting training/testing data so that
        # trials are selected together (since formatting flattens time pts)
       
        # loop over models
        for model in models:
            if verbosity > 1:
                print "Processing %s" % (model)
            # do gam, get prediction
            rout = rgam(data_train, data_test, model, True, family)
            gam_fold_model = GAMFoldModel(rout, data_train[:,1], \
                count[test].shape, test)
            
            # put in container         
            gam_unit[model, icv] = gam_fold_model
    
    # now fold gam_unit.pred and gam_unit.actual back into tasks/repeats
    gam_unit.fold_repeats(bnd)
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
    bdict = {}
    
    if '(Intercept)' in pname:
        bdict['k'] = p[:,np.flatnonzero(pname == '(Intercept)')].squeeze()

    # identify velocity or direction
    # assumes only one of 'v' or 't' is specified
    if np.any(np.in1d(['dx', 's(t):dx.1'], pname)):
        q = 'd'
    elif np.any(np.in1d(['vx', 's(t):vx.1'], pname)):
        q = 'v'
    else:
        q = None

    # get constant terms, if present
    if '%sx' % q in pname:
        assert('%sy' % q in pname)
        assert('%sz' % q in pname)
        qx = p[:,np.flatnonzero(pname == '%sx' % (q))].squeeze()
        qy = p[:,np.flatnonzero(pname == '%sy' % (q))].squeeze()
        qz = p[:,np.flatnonzero(pname == '%sz' % (q))].squeeze()
        bdict[q] = np.array([qx, qy, qz])

    # get dynamic terms, if present
    if 's(t):%sx.1' % q in pname:
        # identify number of bins
        pattern = 's(t):%sx' % (q)
        nbin = np.sum([pattern in x for x in pname])

        darr = np.zeros((3, ncv, nbin))
        for i, dim in enumerate(dims):
            first, last = ['s(t):%s%s.%s' % (q, dim, j) \
                               for (j) in [1, nbin]]
            fidx = np.flatnonzero(pname == first)[0]
            lidx = np.flatnonzero(pname == last)[0]
            darr[i] = p[:, fidx:lidx + 1]
        bdict[q + 't'] = darr
            
    if 'px' in pname:
        # get preferred position
        px = p[:,np.flatnonzero(pname == 'px')].squeeze()
        py = p[:,np.flatnonzero(pname == 'py')].squeeze()
        pz = p[:,np.flatnonzero(pname == 'pz')].squeeze()
        bdict['p'] = np.array([px,py,pz]) # has shape ndim, ncv
        
    if 'sp' in pname:
        bdict['s'] = p[:, np.flatnonzero(pname == 'sp')].squeeze()

    return bdict

def _get_nparam(model, nbin):
    '''
    Return the number of parameters involved in `model`.
    '''
    nparam = 0
    if 'k' in model:
        nparam += 1
    if ('d' in model) or ('v' in model):
        # just to be sure - these should be mutually exclusive
        assert('q' not in model)
        if 'X' not in model:
            nparam += 3
        else:
            nparam += 3 * nbin
    if 'q' in model:
        assert('X' in model)
        nparam += 3 * (nbin + 1)
    if 'p' in model:
        nparam += 3
    if 's' in model:
        nparam += 1
    return nparam

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

#~ def calc_sample_loglik(gam_unit, family='poisson'):
    #~ '''
    #~ NOT UPDATED TO WORK WITH GAM_PLUS VERSION
    #~ 
    #~ Calculate log likelihood loss function for Poisson or Gaussian
    #~ distributed data.
#~ 
    #~ Log likelihood is defined as:
#~ 
    #~ .. math:: L(Y, \theta(X)) = -2 \cdot \log \mathtext{Pr}_{\theta(X)}(Y)
    #~ 
    #~ where :math:`\theta(X)` is the prediction and :math:`Y` is the actual data.
    #~ From Friedman, Tibshirani, and Hastie, 2nd ed, 5th print, eq. 7.8.
    #~ This is the probability of seeing the data, given the prediction.
    #~ 
    #~ For Poisson, :math:`\mathtext{Pr}_{\theta(X)}(Y)` is given by 
    #~ :math:`pmf(Y,f(X))` and for gaussian, by :math:`pdf(Y, f(X))`
    #~ '''
    #~ assert family in ['poisson', 'gaussian']
    #~ 
    #~ if family == 'poisson':
        #~ # only needs mean shape parameter
        #~ Pr = -2 * stats.poisson.logpmf(gam_unit.actual, gam_unit.pred)
    #~ else:
        #~ # normal needs means and variance
        #~ # shape handling here calculates average across repeats,
        #~ # but keeps that dimension for broadcasting purposes
        #~ pred_std = stats.nanstd(gam_unit.pred, axis=2)[:,:,None]
        #~ Pr = -2 * stats.norm.logpdf(gam_unit.actual, gam_unit.pred, pred_std)
    #~ 
    #~ # Pr now has shape (nmod, ntask, nrep, nunit, nbin)
    #~ # get one number per model...
    #~ return get_mean_sem_of_samples(Pr)
    
#~ def calc_loglik(gam_unit, family='poisson'):
    #~ '''
    #~ NOT UPDATED TO WORK WITH GAM_PLUS VERSION
    #~ 
    #~ Calculate log likelihood for Poisson or Gaussian distributed data.
    #~ '''
    #~ assert family in ['poisson', 'gaussian']
    #~ 
    #~ if family == 'poisson':
        #~ # only needs mean shape parameter
        #~ Pr = stats.poisson.logpmf(gam_unit.actual, gam_unit.pred)
    #~ else:
        #~ # normal needs means and variance
        #~ # shape handling here calculates average across repeats,
        #~ # but keeps that dimension for broadcasting purposes
        #~ pred_std = stats.nanstd(gam_unit.pred, axis=2)[:,:,None]
        #~ Pr = stats.norm.logpdf(gam_unit.actual, gam_unit.pred, pred_std)
        #~ 
    #~ # Pr now has shape (nmod, ntask, nrep, nunit, nbin)
    #~ # get one number per model...
    #~ return get_sum_of_samples(Pr)
    
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

    tau = {}
    P   = {}
    for k, v in gam_unit.fits.iteritems():
        if not average:
            pred_flat = v.pred.flatten()[~nans]
        else:
            pred_flat = stats.nanmean(v.pred, axis=1).flatten()
        tau[k], P[k] = stats.kendalltau(act_flat, pred_flat)
    return tau, P

#~ def calc_mse(gam_unit):
    #~ '''
    #~ NOT UPDATED TO WORK WITH GAM_PLUS VERSION
    #~ 
    #~ Calculate the Mean Squared Error (MSE) of a prediction.
    #~ 
    #~ Is the mean across samples of the squared difference between actual and
    #~ predicted values.
    #~ 
    #~ .. math:: \text{MSE} = <(Y_{\mathrm{act} - Y_{pred})^2>_\mathrm{n}
    #~ '''
    #~ actual = gam_unit.actual
    #~ pred = gam_unit.pred
    #~ se = (actual[None] - pred)**2
    #~ nmodel = se.shape[0]
    #~ se = np.reshape(se, [nmodel, np.prod(se.shape[1:])])
    #~ mse = stats.nanmean(se, axis=1)
    #~ return mse

#~ def calc_nagelkerkes_r2(gam_unit):
    #~ '''
    #~ NOT UPDATED TO WORK WITH GAM_PLUS VERSION
    #~ 
    #~ Calculate Nagelkerke's R^2 [1].
    #~ 
    #~ Nagelkerke NJD (1991). A Note on a General Definition of 
    #~ the Coefficient of Determination. Biometrika 78, 691-692.
    #~ 
    #~ Notes
    #~ -----
    #~ 
    #~ Nagelkerke's R2 is given by:
    #~ 
    #~ .. math:: R^2 = \frac{1 - \frac{L(0)}{L(\hat\theta)^{2/n}}}{1-(L(0))^{2/n}}
    #~ '''
    #~ # check last model is intercept-only
    #~ assert(gam_unit.coef_names[-1][0] == '(Intercept)')
    #~ assert(np.all(gam_unit.coef_names[-1][1:] == ''))
    #~ 
    #~ logl0 = gam_unit.logl[-1] # log likelihood for null model, shape 10
    #~ loglm = gam_unit.logl     # log likelihood for other models, shape (17,10)
    #~ 
    #~ # for some reason, actual # non-nans > pred # non-nans
    #~ n  = (~np.isnan(gam_unit.pred[0])).sum().astype(float)
#~ 
    #~ # do most of calculations in log space to avoid numeric over/underflows
    #~ numerator   = 1 - np.exp(2/n * (logl0 - loglm))
    #~ denominator = 1 - np.exp(2/n * logl0)
    #~ R2s = numerator / denominator
    #~ 
    #~ # take mean across cross-validations
    #~ # should really use a weighted mean, 
    #~ # taking into account size of each CV
    #~ return np.mean(R2s, axis=1)
    
    

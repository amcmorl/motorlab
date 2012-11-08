import numpy as np
from warnings import warn
from copy import deepcopy

class BinnedData:
    '''
    Object for returning PSTH and positional data

    Attributes
    ----------
    count : ndarray or None
      binned spike counts, sorted by task and repeats
      shape (ntask, nrep, nunit, nbin)
    bin_edges : ndarray
      bin edge times, sorted as per PSTHs
      shape (ntask, nrep, nbin + 1)
    pos : ndarray
      position data at bin edges, sorted as per PSTHs
      shape (ntask, nrep, nbin + 1)
    tasks : ndarray
      start and end positions of each task
      shape (ntask, 6)
    unit_names : ndarray
      names of each unit, shape (nunit)
    lags : ndarray
      lags of each unit, shape (nunit)
    align : string
      align method used
    align_start_bins, align_end_bins : ndarray, optional
      start and end points for trial alignment, if calculated
    scaled_spikes : list of list, optional
      spike times, scaled as per trials
      
    Parameters
    ----------
    bin_edges : ndarray
      bin edge times, sorted as per PSTHs
      shape (ntask, nrep, nbin + 1)
    pos : ndarray
      position data at bin edges, sorted as per PSTHs
      shape (ntask, nrep, nbin + 1)
    '''
    full_output = False
    
    def __init__(self,
                 bin_edges,
                 pos,
                 tasks,
                 unit_names,
                 align,
                 count=None,
                 unbiased_rate=None,
                 lags=None,
                 align_start_bins=None,
                 align_end_bins=None,
                 scaled_spikes=None,
                 files=None):
        '''
        Anything here?
        '''
        # should have some input testing
        
        self.bin_edges = bin_edges
        self.pos = pos

        self.count = count # assign None if not provided
        self.unbiased_rate = unbiased_rate

        #ntask, nrep, nbin = bin_edges.shape
        self.tasks = tasks
        self.unit_names = unit_names
        self.lags = lags
        self.align = align

        self.align_start_bins = align_start_bins
        self.align_end_bins = align_end_bins
        self.scaled_spikes = scaled_spikes
        self.files = files
            
    def get_notnans(self):
        '''
        Find trials in flattened arrays that don't have any nans.

        Returns
        -------
        notnans : ndarray
          bool array of non-nan-containing trials, shape (ntask * nrep)
        '''
        ntask, nrep, nbin = self.pos.shape[0:3]
        pos_flat = self.pos.reshape(ntask * nrep, nbin, 3)
        return ~np.any(np.any(np.isnan(pos_flat), axis=1), axis=1)
            
    def get_count_flat(self, with_nans=True, squeeze=False):
        '''
        Returns flattened version of count, with all trials iterated
        in first dimension.
        
        Parameters
        ----------
        with_nans : bool
          include trials with nans or not
        squeeze : bool
          remove single dataset dimension if applicable

        Returns
        -------
        count_flat : ndarray
          flattened PSTHs, shape (ntask * nrep, nunit, nbin)
        '''
        if self.count == None:
            raise ValueError('count component does not exist.')
        
        ntask, nrep = self.pos.shape[0:2]
        ndset, nbin = self.count.shape[2:]
        if not with_nans:
            notnans = self.get_notnans()
        else:
            notnans = slice(None)
        res  = self.count.reshape(ntask * nrep, ndset, nbin)[notnans]
        if squeeze:
            return res.squeeze()
        else:
            return res

    def get_unbiased_rate_flat(self, with_nans=True, squeeze=False):
        '''
        Returns flattened version of unbiased_rate, with all trials iterated
        in first dimension.
        
        Parameters
        ----------
        with_nans : bool
          include trials with nans or not
        squeeze : bool
          remove single dataset dimension if applicable

        Returns
        -------
        count_flat : ndarray
          flattened PSTHs, shape (ntask * nrep, nunit, nbin)
        '''
        if self.unbiased_rate == None:
            raise ValueError('unbiased_rates component does not exist.')
        
        ntask, nrep = self.pos.shape[0:2]
        ndset, nbin = self.unbiased_rate.shape[2:]
        if not with_nans:
            notnans = self.get_notnans()
        else:
            notnans = slice(None)
        res  = self.unbiased_rate.reshape(ntask * nrep, ndset, nbin)[notnans]
        if squeeze:
            return res.squeeze()
        else:
            return res
    
    def get_rates(self, squeeze=False):    
        '''
        Returns firing rates (count / window durations).
        '''
        if self.count == None:
            raise ValueError('count component does not exist.')
        ntask, nrep = self.pos.shape[0:2]
        ndset, nbin = self.count.shape[2:]

        count = self.count
        window = np.diff(self.bin_edges, axis=-1)
        window = np.tile(window[:,:,None], (1, 1, ndset, 1))
        rate = count / window        
        
        if squeeze:
            return rate.squeeze()
        else:
            return rate
        
    def get_rates_flat(self, with_nans=True, squeeze=False):
        '''
        Returns flattened version of firing rates (count / window durations),
        with all trials iterated in first dimension.
        
        Parameters
        ----------
        with_nans : bool
          include trials with nans or not
        squeeze : bool
          remove single dataset dimension if applicable

        Returns
        -------
        PSTHs_flat : ndarray
          flattened PSTHs, shape (ntask * nrep, nunit, nbin)
        '''
        rate = self.get_rates(squeeze=squeeze)
        
        ntask, nrep = self.pos.shape[0:2]
        ndset, nbin = self.count.shape[2:]
        if not with_nans:
            notnans = self.get_notnans()
        else:
            notnans = slice(None)
        flatrate  = rate.reshape(ntask * nrep, ndset, nbin)[notnans]
        
        if squeeze:
            return flatrate.squeeze()
        else:
            return flatrate
        
    def get_pos_flat(self, with_nans=True):
        '''
        Returns flattened version of pos, with all trials iterated
        in first dimension.
        
        Parameters
        ----------
        with_nans : bool
          include nan-containing trials
        
        Returns
        -------
        flat_pos : ndarray
          shape (ntask * nrep, nbin + 1, 3)
        '''
        ntask, nrep, nbin = self.pos.shape[0:3]
        pos_flat = self.pos.reshape(ntask * nrep, nbin, 3)
        if not with_nans:
            notnans = self.get_notnans()
        else:
            notnans = slice(None)
        return pos_flat[notnans]

    def get_bin_edges_flat(self, with_nans=True):
        '''
        Returns flattened version of bin_edges, with all trials iterated
        in first dimension.
        
        Parameters
        ----------
        with_nans : bool
          include nan-containing trials
        
        Returns
        -------
        flat_bin_edges : ndarray
          shape (ntask * nrep, nbin + 1)
        '''
        ntask, nrep, nbin = self.bin_edges.shape[0:3]
        bin_edges_flat = self.bin_edges.reshape(ntask * nrep, nbin)
        if not with_nans:
            notnans = self.get_notnans()
        else:
            notnans = slice(None)
        return bin_edges_flat[notnans]

    def get_all_flat(self, with_nans=True):
        '''Get flattened pos, firing rates and times with or without nans.

        Parameters
        ----------
        with_nans : bool, optional
          include trials containing nans

        Returns
        -------
        time, pos, frs : ndarray
        '''
        if self.count == None:
            raise ValueError('count component does not exist.')
        
        time = self.get_bin_edges_flat(with_nans=with_nans)
        pos = self.get_pos_flat(with_nans=with_nans)
        count = self.get_count_flat(with_nans=with_nans)
        return count, pos, time

    def get_for_glm(self, dset_idx=None):
        '''Get count, pos, and time in a format suitable for
        passing to regress.

        Parameters
        ----------
        unit_num : int
          index of dataset to select

        Returns
        -------
        count, pos, time : ndarray
        '''
        if self.count == None:
            raise ValueError('count component does not exist.')
        
        if dset_idx == None:
            print "Defaulting to dataset 0"
            dset_idx = 0

        time = self.get_bin_edges_flat(with_nans=False)
        pos = self.get_pos_flat(with_nans=False)
        count = self.get_count_flat(with_nans=False)[:, dset_idx]
        return count, pos, time

    def get_for_GLM(self, dset_idx=None):
        warn(DeprecationWarning("Use get_for_glm instead."))
        return self.get_for_glm(dset_idx)
    
    def get_for_regress(self, dset_idx=None, use_unbiased=True):
        '''Get rate, pos, and time in a format suitable for
        passing to regress.

        Parameters
        ----------
        unit_num : int
          index of dataset to select

        Returns
        -------
        rate, pos, time : ndarray
        '''
        if use_unbiased:
            if self.unbiased_rates == None:
                raise ValueError('unbiased rates component does not exist.')
        
        if dset_idx == None:
            print "Defaulting to dataset 0"
            dset_idx = 0

        time = self.get_bin_edges_flat(with_nans=False)
        pos = self.get_pos_flat(with_nans=False)
        if use_unbiased:
            rate = self.get_unbiased_rates_flat(with_nans=False)[:,dset_idx]
        else:
            rate = self.get_rates_flat(with_nans=False)[:,dset_idx]
        return rate, pos, time
    
    def has_flat(self):
        '''
        DEPRECATED - use get_<x>_flat methods instead of flat attributes.
        
        Returns
        -------
        has_flat : bool
          if instance has flat attributes
        '''
        warn(DeprecationWarning("Use get_<x>_flat instead of flat attributes."))
        if 'PSTHs_flat' in self.__dict__.keys():
            assert 'pos_flat' in self.__dict__.keys()
            # check pos is also present
            return True
        else:
            return False
    
    def set_count_from_flat(self, new_count_flat):
        '''
        Sets PSTHs attribute to new values from a flat array.
        
        Parameters
        ----------
        new_PSTHs_flat : ndarray
          new firing rates, must have same shape as old ones
          shape (ntask * nrep, ndset, nbin)
        '''
        if self.count == None:
            # take shape values from pos
            ntask, nrep, old_nbin, ndim = self.pos.shape
            ntrial, ndset, nbin = new_count_flat.shape
            if (ntask * nrep != ntrial) | (nbin != old_nbin - 1):
                raise ValueError("shape is not compatible with kinematic data")
        else:
            ntask, nrep, ndset, nbin = self.count.shape
            # must take on same shape as old PSTHs
            assert new_count_flat.shape == (ntask * nrep, ndset, nbin)
        new_count = new_count_flat.reshape(ntask, nrep, ndset, nbin)  
        self.count = new_count

    def set_count(self, new_count):
        '''
        Sets PSTHs attribute to new values.
        
        Parameters
        ----------
        new_PSTHs : ndarray
          new rates to set, shape must equal old rates
          shape (ntask, nrep, ndset, nbin)
        '''
        if self.count != None:
            assert new_count.shape == self.count.shape
        self.count = new_count
        #deprecated
        #if self.has_flat():
        #    self.count_flat = self.get_count_flat()
            
    def keep_only(self, dsets):
        '''
        Trims count to only the dsets indexed in `dsets`.
        
        Parameters
        ----------
        dsets : array_like
          indices of datasets to keep, shape (n,)
        '''
        if self.count == None:
            raise ValueError('count component does not exist.')
        
        self.count = self.count[:,:,dsets]
        if 'count_flat' in self.__dict__.keys():
            self.count_flat = self.count_flat[:,dsets]

    def save(self, file):
        '''
        Writes self to an npz file for future use.

        Parameters
        ----------
        file : str or file
          filename or file-like object as per np.savez; see its docstring

        Returns
        -------
        None

        Notes
        -----
        Doesn't save scaled spikes attributes.
        '''
        data = {'count'      : self.count,
                'bin_edges'  : self.bin_edges,
                'pos'        : self.pos,
                'tasks'      : self.tasks,
                'unit_names' : self.unit_names,
                'lags'       : self.lags,
                'align'      : self.align,
                'files'      : self.files}
        if self.__dict__.has_key('align_start_bins'):
            data['align_start_bins'] = self.align_start_bins
        if self.__dict__.has_key('align_end_bins'):
            data['align_end_bins'] = self.align_end_bins
        if self.unbiased_rate != None:
            data['unbiased_rate'] = self.unbiased_rate
        np.savez(file, **data)

    def ensure_flat_inplace(self):
        '''
        DEPRECATED - Use get_<x>_flat instead of flat attributes.
        
        Make sure instance has flattened versions of sorted arrays. If they
        don't exist, create them.
        '''
        warn(DeprecationWarning("Use get_<x>_flat instead of flat attributes."))
        if not (('pos_flat' in self.__dict__.keys()) and \
                    ('count_flat' in self.__dict__.keys())):
            n_tasks, n_reps = self.pos.shape[0:2]
            n_dsets, n_bins = self.count.shape[2:]
            self.pos_flat = self.pos.reshape(n_tasks * n_reps, n_bins + 1, 3)
            self.count_flat = self.count.reshape(n_tasks * n_reps,
                                                 n_dsets, n_bins)
                                                 
    def __add__(self, other):
        '''
        Add two bnds together.
        
        Combines count along `unit` axis, appends lags and unit_names.
        Checks bin_edges, pos, tasks, files, align are the same.
        are the same. Doesn't handle unbiased rates for now.
        '''
        # check type
        if type(other) != type(self):
            raise ValueError('Only two BinnedData instances can be added.')
        
        # check bin_edges
        a = self.bin_edges
        b = other.bin_edges        
        if not np.all(a[~np.isnan(a)] == b[~np.isnan(b)]):
            raise ValueError('bin_edges are not the same')
            
        # check pos
        a = self.pos
        b = other.pos
        if not np.all(a[~np.isnan(a)] == b[~np.isnan(b)]):
            raise ValueError('pos are not the same')
        
        # check tasks
        a = self.tasks
        b = other.tasks
        if not np.all(a[~np.isnan(a)] == b[~np.isnan(b)]):
            raise ValueError('tasks are not the same')
        
        # check align
        if not np.all(self.align == other.align):
            raise ValueError('align is not the same')
            
        # check files
        if not np.all(self.files == other.files):
            raise ValueError('files are not the same')
            
        # combine stuff to be combined
        a = self
        b = other
        new_count = np.concatenate([a.count, b.count], axis=2)
        new_unames = np.concatenate([a.unit_names, b.unit_names], axis=0)
        new_lags = np.concatenate([a.lags, b.lags], axis=0)
        
        return BinnedData(self.bin_edges, self.pos, self.tasks,
                          new_unames, self.align, count=new_count,
                          lags=new_lags, files=self.files)
                          
    def copy(self):
        '''
        Make a copy of myself and return it.
        '''
        return deepcopy(self)

def load_binned_data(file):
    '''
    Loads a BinnedData instance from a file (saved by SortedData.save)

    Parameters
    ----------
    file : str or file
      filename or object from which to load data
    '''
    bdf = np.load(file)
    bnd = BinnedData(bdf['bin_edges'],
                     bdf['pos'],
                     bdf['tasks'],
                     bdf['unit_names'],
                     bdf['align'])
    if 'count' in bdf.files:
        bnd.count = bdf['count']
    if 'lags' in bdf.files:
        bnd.lags = bdf['lags']
    if 'unbiased_rate' in bdf.files:
        bnd.unbiased_rate = bdf['unbiased_rate']
    if ('align_start_bins' in bdf.files) & ('align_end_bins' in bdf.files):
        bnd.align_start_bins = bdf['align_start_bins']
        bnd.align_end_bins = bdf['align_end_bins']
    if 'files' in bdf.files:
        bnd.files = bdf['files']
    return bnd

def has(obj, item):
    return item in obj.__dict__.keys()
    
def _estimate_count_from_rate(rate, bin_edges):
    '''
    Convert a `rate` (in Hz) to a count, given window edges `bin_edges`.
    '''
    assert rate.shape[-1] == bin_edges.shape[-1] -1
    
    window = np.diff(bin_edges, axis=-1)
    
    if np.rank(rate) == np.rank(bin_edges) + 1:
        # add dimension at -2 if there is one more dim in rate
        window = window[...,None,:]
    return rate * window

def _make_count_from_rate(rate, bin_edges):
    '''
    Make array of count values (ints (but stored in float arr)) from Poisson rvs with mean given by rate.

    Need to take into account window size.

    rate : ndarray
      shape is same as count will have
    '''
    assert rate.shape[-1] == bin_edges.shape[-1] -1
    
    est_count = _estimate_count_from_rate(rate, bin_edges)

    count = np.zeros_like(est_count)

    # False for nans and negatives
    nonneg = est_count >= 0
    count[nonneg] = np.random.poisson(est_count[nonneg])

    # also need to keep track of original nans
    isnan = np.isnan(est_count)
    count[isnan] = np.nan
    
    return count

def make_bnd_with_count_from_rate(rate, bnd):
    '''
    Make a copy of `bnd` with the count taken as a Poisson RVS based on `rate`.

    Parameters
    ----------
    rate : ndarray
      shape = bnd.count.shape
    bnd : BinnedData
      binned data to base return value off
    '''
    assert rate.shape == bnd.count.shape
    count = _make_count_from_rate(rate, bnd.bin_edges)
    bndc = bnd.copy()
    bndc.set_count(count)
    return bndc

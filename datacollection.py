import numpy as np
from motorlab.data_files import CenOut_VR_RTMA_10_File, \
    CenOut_3d_data_21_File, versions
import motorlab.tuning.util as tc_util
from warnings import warn
from scipy.interpolate import splprep, splev
from amcmorl_py_tools.vecgeom import norm, unitvec
from motorlab.binned_data import BinnedData
import motorlab.kinematics as kin

'''
want to have something that:
  reads in files
  generates alignment times and binned positional data
  extracts spike times and bins as required
'''
align_types = ["all", "speed", "hold", "323 simple", "323 movement"]

def _add_to_array(array, to_add):
    assert (type(array) == type(None)) | (type(array) == type(np.array(0)))
    if type(array) != type(None):
        assert (np.rank(array) == np.rank(to_add))
        
    if array == None:
        array = to_add
    else:
        array = np.hstack((array, to_add))
    return array

def _close(x,y, atol=1.e-8, rtol=1.e-5):
    return np.less_equal(np.absolute(x - y), atol + rtol * np.absolute(y))

def _interpolate_position(positions, bins, s=0., k=5, nest=-1):
    '''
    s : float
    smoothness
    k : int
    spline order
    nest : int
    est. of number of knots, -1 = maximal
    '''
    x,y,z = [q[0] for q in zip(positions.T[:-1])]
    t = positions.T[-1]
    tckp, u = splprep([x,y,z], u=t, s=s, k=k, nest=nest)
    xnew, ynew, znew = splev(bins, tckp)
    return np.vstack((xnew,ynew,znew)).T

class Unit:
    '''
    Container for a unit, spikes and name.

    Attributes
    ----------
    parent : DataCollection
      collection of raw data to which this units spikes belong
    unit_name : string
      name of unit, channel and sort, e.g. 'Unit001_1'
    lag : float
      lag time between neural and movement data
      positive lag implies neural event precedes kinematic

    
    Parameters
    ----------
    unit_name : string
      see Attributes
    lag : float
      see Attributes
    datacollection : DataCollection
      see Attributes
    '''
    def __init__(self, unit_name, lag, datacollection):
        self.parent = datacollection
        self.unit_name = unit_name
        self.lag = lag
        self.spikes = []
        self.get_spikes()

        # check that the number of trials is consistent with other data
        assert len(self.spikes) == len(self.parent.HoldAStart)

    def get_full_name(self):
        '''
        Return the full name of the unit derived from unit name and lag.
        
        Returns
        -------
        full_name : string
          full name of unit including lag, in format "Unit001_1_100ms"
        '''
        return self.unit_name + '_%dms' % int(self.lag * 1e3)
        
    def get_spikes(self):
        '''
        Populates spike data from .mat file
        '''
        # pick correct file handling routine
        if self.parent.version == 'VR_RTMA_1.0':
            opener = CenOut_VR_RTMA_10_File
        elif self.parent.version == '3d_data_2.1':
            opener = CenOut_3d_data_21_File
        else:
            raise ValueError("Not a recognized format.")

        for file_name in self.parent.files:
            # open file and grab sorted spikes
            exp_file = opener(file_name)
            file_spikes = exp_file.sort_spikes(self.unit_name, lag=self.lag)
            self.spikes.extend(file_spikes)
            
class DataCollection:
    '''
    A collection of raw data collated from a number of .mat files. One
    DataCollection contains one set of positions and trial times,
    and zero or more sets of spike times for each channel of interest.
    
    Parameters
    ----------
    files : list of string
      list of filenames to load

    Attributes
    ----------
    files : list of string
      list of filenames to load
    units : list of Units
      list of Unit instances
    positions : list of ndarray
      list of arrays of position data, len ntrials, each shape (nsample, 4)
    HoldAStart, HoldAFinish : ndarray
      enter target and disappearance of Hold A target, len ntrials
      all times are relative to PlexonTrialTime
    HoldBStart, HoldBFinish : ndarray
      enter target and disappearance of Hold B target, len ntrials
    StartPos, TargetPos : ndarray
      starting and target positions of each trial, shape (ntrial, 3)
    ReactionFinish : ndarray
      time when cursor leaves position previously occupied by Hold A target
      shape (ntrial,)
    PlexonTrialTime : ndarray
      time, relative to start of whole file, of this trial
      all other times are relative to these times
      shape (ntrial,)
    '''
    def __init__(self, files):
        self.files = []
        self.units = []
        self.positions = []
        self.HoldAStart = self.HoldAFinish = None
        self.HoldBStart = self.HoldBFinish = None
        self.TargetPos  = self.StartPos    = None
        self.ReactionFinish  = None
        self.PlexonTrialTime = None
        
        for file in files:
            self.add_file(file)
        self.collate_trials()

    def add_file(self, file_name, version='VR_RTMA_1.0'):
        """Adds a file's kinematic data to the cell.

        Parameters
        ----------
        file_name : string
          full name of file to load
        version : string
          one of 'VR_RTMA_1.0'
        """
        assert type(file_name) == str
        assert (version in versions) | (version in range(len(versions)))

        if type(version) == int:
            version = versions[version]
        self.version = version
        self.files.append(file_name)
        
        if version == 'VR_RTMA_1.0':
            exp_file = CenOut_VR_RTMA_10_File(file_name)
        #elif version == '3d_data_2.1':
        #    exp_file = CenOut_3d_data_21_File(file_name)
        else:
            raise ValueError("Not a recognized format.")
            
        # extract bits of information from files needed in subsequent analysis
        # should all be in same order in long arrays (shape -> (n_trials,))

        file_kins = exp_file.get_kinematics()
        self.positions.extend(file_kins)
        
        # gives a list of arrays of slightly different lengths,
        # of 3d positions - shape [(3, n_samples),...]

        self.HoldAStart = _add_to_array(self.HoldAStart, exp_file.HoldAStart)
        self.HoldAFinish = _add_to_array(self.HoldAFinish, exp_file.HoldAFinish)
        self.HoldBStart = _add_to_array(self.HoldBStart, exp_file.HoldBStart)
        self.HoldBFinish = _add_to_array(self.HoldBFinish, exp_file.HoldBFinish)
        self.ReactionFinish = _add_to_array(self.ReactionFinish,
                                               exp_file.ReactionFinish)
        self.PlexonTrialTime = _add_to_array(self.PlexonTrialTime,
                                            exp_file.PlexonTrialTime)

        # _add_to_array only handles 1d arrays
        if self.StartPos == None:
            self.StartPos = exp_file.StartPos
        else:
            self.StartPos = np.concatenate((self.StartPos,
                                            exp_file.StartPos),
                                           axis=0)
        if self.TargetPos == None:
            self.TargetPos = exp_file.TargetPos
        else:
            self.TargetPos = np.concatenate((self.TargetPos,
                                             exp_file.TargetPos),
                                            axis=0)

    def collate_trials(self):
        '''Perform consolidation operations once all files have been added.
        '''
        self.tasks = sort_unique_tasks(self.StartPos, self.TargetPos)

    def get_uniq_name(self):
        '''
        Gets the unique name, derived from full names of all units.
        
        Returns
        -------
        unique_name : string
          unique name of data collection
        '''
        return '_'.join([unit.get_full_name() for unit in self.units])

    def get_unit_names(self):
        '''
        Returns
        -------
        name_list : list of string
          list of unit full names
        '''
        return [unit.get_full_name() for unit in self.units]

    # movement stuff ----------------------------------------------------------

    def calc_movement_times(self, threshold=0.15,
                            earlymax_limit=0.33,
                            latemax_limit=0.75,
                            full_output=False):
        '''Finds the times, from spike 0, at which speed crosses
        `threshold` * 100% * maximum speed.

        Parameters
        ----------
        threshold : float
            proportion of maximum speed to take as the start and finish of
            movement period
        earlymax_limit : float
            
        Returns
        -------
        movement_start_times : array_like, shape (n_trials,2)
        movement_finish_times : array_like, shape (n_trials,2)
            times, relative to spike time 0 (PlexonTrialTime),
            of movement start and finish times
        [speeds : list of 1d arrays]
            
        [times : list of 1d arrays]

        Notes
        -----
        times are relative to PlexonTrialTime
        '''
        assert type(threshold) == float
        assert type(earlymax_limit) == float
        assert type(latemax_limit) == float
        assert type(full_output) == bool
        
        n_trials = len(self.positions)
        self.movement_start = np.empty(n_trials) + np.nan
        self.movement_stop = np.empty(n_trials) + np.nan
        if full_output:
            times = []
            speeds = []
        for i, trial in enumerate(self.positions):
            t = trial[:,3]
            #spd = kin.get_speed(trial[:,0:3], t, tax=0, spax=-1)

            spd = self.get_projected_speed(i)
            
            #dt = np.diff(t)
            #dp = np.diff(trial[:,0:3], axis=0)
            #vel = dp / dt[...,None]
            #spd = norm(vel, axis=1)

            prelim = spd.size * earlymax_limit
            postlim = spd.size * latemax_limit
            max_pt = np.argmax(spd[prelim:postlim]) + prelim
            nspd = spd / spd[max_pt]
            
            if full_output:
                times.append(t)
                speeds.append(nspd)

            cross = np.diff((nspd > threshold).astype(int))
            # most stringent requirements would be one up before max_pt
            # and one down after max_pt
            # I will opt for at least one up before max_pt (and take last)
            # and first of however many downs after max_pt
            up = np.flatnonzero(cross == 1)
            valid_up = up[up < max_pt]
            if valid_up.size > 0:
                upt = valid_up[-1]
                tup = tc_util.twopt_interp(t[upt], t[upt + 1],
                                             nspd[upt], nspd[upt + 1],
                                           threshold)
            else:
                warn('Cannot find up point in trial %d.' % (i))
                tup = np.nan
                
            down = np.flatnonzero(cross == -1)
            #if i == 20: 1/0.
            valid_down = down[down > max_pt] 
            if valid_down.size > 0:
                dpt = valid_down[0]
                #... -1 to get pt before crossing threshold
                tdown = tc_util.twopt_interp(t[dpt], t[dpt + 1],
                                             nspd[dpt], nspd[dpt + 1],
                                             threshold)
            else:
                warn('Cannot find down point in trial %d.' % (i))
                tdown = np.nan

            if ~np.isnan(tup) and ~np.isnan(tdown):
                self.movement_start[i] = tup
                self.movement_stop[i] = tdown
                assert tup < tdown
                assert tup > t[0]
                assert tdown < t[-1]
            else:
                self.movement_start[i] = np.nan
                self.movement_stop[i] = np.nan
        return self.movement_start, self.movement_stop

    def get_projected_speed(self, tno):
        '''calculate speed of trial number `tno`, projected along task direction

        Parameters
        ----------
        tno : int
          index of trial to calculate

        Returns
        -------
        proj : ndarray
          projected speed values, in m/s
          
        Notes
        -----
        we have to do them one-at-a-time because positions has different
        numbers of elements in each trial.        
        '''
        
        trial = self.positions[tno]
        pos = trial[:,0:3]
        time = trial[:,3]
        vel = kin.get_vel(pos, time, tax=0, spax=-1)
        task_dir = unitvec(self.TargetPos[tno] - self.StartPos[tno])
        proj = np.dot(vel, task_dir)
        return proj
        
    def get_movement_times(self):
        '''
        Returns fitted movement start and finish times, when speed crosses
        a certain threshold percentage of maximum.
        
        Returns
        -------
        movement_start, movement_stop : ndarray
          start and stop times calculated by calc_movement_times
        '''
        if not 'movement_start' in self.__dict__.keys():
            self.calc_movement_times()
        return self.movement_start, self.movement_stop
        
    def calc_max_repeats(self):
        '''
        Returns maximum number of repeats of any one task in the dataset.
        Determines the shape of the sorted data arrays.
        
        Returns
        -------
        max_reps : int
          maximum number of repeats of any one direction in center out  
        '''
        n_trials = self.HoldAStart.size
        n_dirs = self.tasks.shape[0]
        dirs_count = np.zeros(n_dirs, dtype=int)
        # step over trials
        for i in xrange(n_trials):
            dir_idx = tc_util.get_task_idx(self.StartPos[i],
                                           self.TargetPos[i],
                                           self.tasks)
            dirs_count[dir_idx] += 1
        return dirs_count.max()
    
    def calc_clip_time_all(self, verbose=False):
        '''
        Define window start and finish times as beginning and end
        of each trial.
        
        Parameters
        ----------
        verbose : bool, optional
          print alignment scheme name
          
        Returns
        -------
        clip_time_start, clip_time_finish : ndarray
          window start and stop times, relative to PlexonTrialTimes
        '''
        if verbose:
            print "Aligning by all"
        return self.HoldAStart.copy(), self.HoldBFinish.copy()
    
    def calc_clip_time_hold(self, verbose=False):
        '''
        Define window start and finish times as earliest common HoldA time
        and latest common HoldB time after scaling to equal length movement
        times.
        
        Parameters
        ----------
        verbose : bool, optional
          print alignment scheme name
          
        Returns
        -------
        clip_time_start, clip_time_finish : ndarray
          window start and stop times, relative to PlexonTrialTimes
        '''
        if verbose:
            print "Aligning by hold"
            
        align_starts, align_stops = self.get_movement_times()
        window_start, window_stop = self.HoldAStart, self.HoldBFinish
        
        # to normalize all times so time between 15% marks is =,
        # divide by that time duration - st = scaled time
        # in scaled time, time between 15% marks is always 1.
        
        # then find max allowable times which all trials have before and
        # after 15% marks (pre_st.min() and post_st.min())
        # then, for each trial, convert min scaled times to actual times

        bw = align_stops - align_starts
        pre_rt = align_starts - window_start
        pre_st = pre_rt / bw
        pre_rt_cmn = np.nanmin(pre_st) * bw
        
        post_rt = window_stop - align_stops
        post_st = post_rt / bw
        post_rt_cmn = np.nanmin(post_st) * bw

        # only check valid values
        ok = ~np.isnan(bw)
        assert np.all((pre_rt_cmn[ok] < pre_rt[ok]) |
                      _close(pre_rt_cmn[ok], pre_rt[ok]))
        assert np.all((post_rt_cmn[ok] < post_rt[ok]) |
                      _close(post_rt_cmn[ok], post_rt[ok]))

        start_time = align_starts - pre_rt_cmn
        stop_time = align_stops + post_rt_cmn

        _check_times(start_time, stop_time, window_start, window_stop)
        return start_time, stop_time
            
    def calc_clip_time_speed(self, verbose=False):
        '''
        Define window start and finish times as movement epoch start and
        finish times.
        
        Parameters
        ----------
        verbose : bool, optional
          print alignment scheme name
          
        Returns
        -------
        clip_time_start, clip_time_finish : ndarray
          window start and stop times, relative to PlexonTrialTimes
        '''
        if verbose:
            print "Aligning by speed"
        if not 'movement_start' in self.__dict__.keys():
            self.calc_movement_times()

        align_starts = self.movement_start
        align_stops = self.movement_stop
        window_start = align_starts
        start_time = align_starts.copy()
        window_stop = align_stops
        stop_time = align_stops.copy()
        _check_times(start_time, stop_time, window_start, window_stop)
        return start_time, stop_time

    def calc_clip_time_323_movement(self, verbose=False):
        '''
        Define window start and finish times as 30 bins of Hold A,
        20 bins of movement time (defined by movement start and
        finish and HoldBStart), and 30 bins of Hold B.
        
        Parameters
        ----------
        verbose : bool, optional
          print alignment scheme name
          
        Returns
        -------
        clip_time_start, clip_time_finish : ndarray
          window start and stop times, relative to PlexonTrialTimes
        '''
        if verbose:
            print "Aligning by 323_movement"
        if not 'movement_start' in self.__dict__.keys():
            self.calc_movement_times()

        align_starts = self.movement_start
        align_stops = self.movement_stop
        bw = align_stops - align_starts
        pad = 3 * bw / 2.
        start_time = align_starts - pad
        stop_time = align_stops + pad
        window_start = self.HoldAStart
        window_stop = self.HoldBFinish
        _check_times(start_time, stop_time, window_start, window_stop)
        return start_time, stop_time
    
    def calc_clip_time_323_simple(self, verbose=False):
        '''
        Define window start and finish times as 30 bins of Hold A,
        20 bins of movement time (defined by ReactionFinish and
        HoldBStart), and 30 bins of Hold B.
                
        Parameters
        ----------
        verbose : bool, optional
          print alignment scheme name
          
        Returns
        -------
        clip_time_start, clip_time_finish : ndarray
          window start and stop times, relative to PlexonTrialTimes
        '''
        if verbose:
            print "Aligning by 323_simple"
        if not 'movement_start' in self.__dict__.keys():
            self.calc_movement_times()

        align_starts = self.ReactionFinish
        align_stops = self.HoldBStart
        bw = align_stops - align_starts
        pad = 3. * bw / 2.
        start_time = align_starts - pad
        stop_time = align_stops + pad
        window_start = self.HoldAStart
        window_stop = self.HoldBFinish
        v = (start_time > window_start) & (stop_time < window_stop)
        start_time[~v] = np.nan
        stop_time[~v] = np.nan
        _check_times(start_time, stop_time, window_start, window_stop)
        return start_time, stop_time
            
    # spike stuff ------------------------------------------------------------

    def add_units(self, units_list):
        '''
        Add several units to data collection.
        
        Parameters
        ----------
        units_list : list
          each element is (unit_name, lag)
          see DataCollection.add_unit for details
        '''
        for unit, lag in units_list:
            self.add_unit(unit, lag)
    
    def add_unit(self, unit_name, lag):
        '''
        Add a single unit to data collection.
        
        Parameters
        ----------
        unit_name : string
          name of unit in data files, e.g. Unit001_1
        lag : float
          lag time in ms
        '''
        self.units.append(Unit(unit_name, lag, self))

    def _get_limits(self, align):
        if align == 'all':
            align_starts, align_stops = self.HoldAStart, self.HoldBFinish
            bin_starts, bin_stops = self.calc_clip_time_all()
        if align == 'speed':
            align_starts, align_stops = self.calc_movement_times()
            bin_starts, bin_stops = self.calc_clip_time_speed()
        if align == 'hold':
            align_starts, align_stops = self.calc_movement_times()
            bin_starts, bin_stops = self.calc_clip_time_hold()
        if align == '323 simple':
            align_starts, align_stops = self.ReactionFinish, self.HoldBStart
            bin_starts, bin_stops = self.calc_clip_time_323_simple()
        if align == '323 movement':
            align_starts, align_stops = self.calc_movement_times()
            bin_starts, bin_stops = self.calc_clip_time_323_movement()
        return align_starts, align_stops, bin_starts, bin_stops
    
    def make_binned(self, nbin=100,
                    align='speed',
                    verbose=False,
                    do_count=True,
                    do_rate=False):
        '''Constructs aligned PSTHs of spikes in each direction.

    Histograms are constructed with `n_bins`, aligned at increasing
    and decreasing 15% speed points, windowed to include maximum
    common period back to latest HoldAStart and up to earliest
    HoldBFinish.

    Parameters
    ----------
    n_bins : int, default=100
        number of bins
          align : string
            one of 'speed', 'hold', '3:2:3'
        verbose : bool
            print information messages?

        Returns
    -------
        binned : BinnedData instance
          bd.PSTHs.shape = (n_tasks, n_reps, n_dsets, nbins)
          bd.pos.shape = (n_tasks, n_reps, nbins + 1, 3)
    '''
        #assert len(self.units) > 0
        assert type(nbin) == int
        assert align in align_types
        align_starts, align_stops, bin_starts, bin_stops = \
            self._get_limits(align)
        do_spike = (do_count or do_rate)
        
        # sort n_trials trials into n_dirs directions
        # requires calc max_n_repeats per direction to construct array
        max_repeat = self.calc_max_repeats()
        ntrial = bin_starts.size
        ndir = self.tasks.shape[0]
        nunit = len(self.units)

        if (nunit > 0) and do_spike:
            if do_count:
                counts = np.empty((ndir, max_repeat, nunit, nbin)) + np.nan
            if do_rate:
                rates = np.empty((ndir, max_repeat, nunit, nbin)) + np.nan
            else:
                rates = None
        else:
            counts = None
            rates = None
        bin_edges = np.empty((ndir, max_repeat, nbin + 1)) + np.nan
        dirs_count = np.zeros(ndir)
        pos = np.empty((ndir, max_repeat, nbin + 1, 3)) + np.nan

        align_start_bins = None
        align_stop_bins = None

        for i in xrange(ntrial):
            # directions are sorted according to the sorted unique tasks
            
            if np.isnan(bin_starts[i]) or np.isnan(bin_stops[i]):
                # couldn't align trial for some reason- ignore
                if verbose:
                    print "Sort error, trial %d: " \
                        "couldn't find alignment points." \
                        % i
                continue
            
            # put index of this trial into correct direction column
            dir_idx = tc_util.get_task_idx( \
                self.StartPos[i], self.TargetPos[i], self.tasks)

            bins = np.linspace(bin_starts[i], bin_stops[i], nbin + 1, \
                                   endpoint=True)
            
            if (nunit > 0) & do_spike:
                if do_count:
                    trial_count = np.empty((nunit, nbin)) + np.nan
                if do_rate:
                    trial_rate = np.empty((nunit, nbin)) + np.nan
                for i_unit, unit in enumerate(self.units):
                    spikes = np.asarray(unit.spikes[i])
                    if do_count:
                        trial_count[i_unit], _ = np.histogram(spikes, bins=bins)
                    if do_rate:
                        trial_rate[i_unit], _ = tc_util.unbiased_histogram( \
                            np.asarray(unit.spikes[i]), bins=bins)

            # interpolate position
            these_pos = _interpolate_position(self.positions[i], bins)
                
            # calculate (for display purposes later on) position of 15%-acc.
            # in bin numbers - should be constant, but not integer, for each
            bin_width = bins[1] - bins[0]
            this_align_start = (align_starts[i] - bin_starts[i]) / bin_width
            if align_start_bins != None:
                assert np.allclose(this_align_start, align_start_bins)
            align_start_bins = this_align_start
            
            this_align_stop = (align_stops[i] - bin_starts[i]) / bin_width
            if align_stop_bins != None:
                assert np.allclose(this_align_stop, align_stop_bins)
            align_end_bins = this_align_stop

            # tests here
            okay = True
            max_pos_threshold = 0.1 # 10 cm
            min_time_threshold = 0.05 # 50 ms
            
            if np.any(np.abs(these_pos) > max_pos_threshold):
                warn(AlignmentWarning("trial %d: position value too big" % i))
                okay = False
            if (align_stops[i] - align_starts[i]) < min_time_threshold:
                warn(AlignmentWarning("trial %d: movement time too short" % i))
                okay = False
            
            # if passed tests, add to collection
            if okay:
                rep_idx = dirs_count[dir_idx]
                if nunit > 0:
                    if do_count:
                        counts[dir_idx, rep_idx] = trial_count
                    if do_rate:
                        rates[dir_idx, rep_idx] = trial_rate
                bin_edges[dir_idx, rep_idx] = bins
                pos[dir_idx, rep_idx] = these_pos
                dirs_count[dir_idx] += 1
                #... keep track of next row to enter for this direction

        unit_names = [u.unit_name for u in self.units]
        lags = [u.lag for u in self.units]
        return BinnedData(bin_edges, pos, self.tasks, unit_names, align,
                          lags=lags, count=counts, unbiased_rate=rates,
                          align_start_bins=align_start_bins,
                          align_end_bins=align_end_bins,
                          files=self.files)

# utility functions
def sort_unique_tasks(starts, targets):
    '''Return unique items sort in ascending order.

    Parameters
    ----------
    targets : array, shape (m 3)
      target positions for n trials in 3 dimensions

    Returns
    -------
    sorted : array, shape (n, 3)
      unique `targets` sorted in ascending order
    '''
    assert type(starts) == np.ndarray
    assert starts.shape[1] == 3
    assert type(targets) == np.ndarray
    assert targets.shape[1] == 3

    combined = np.concatenate((starts, targets), axis=1)
    #np.savez('urgle.npz', data=combined)

    # explicitly require that `combined` is in C_CONTIGUOUS order
    # which is necessary for view-typing in next step
    # (apparently previous trick of adding 0. doesn't work anymore)
    combined = np.require(combined, requirements='C')

    uniques = np.unique(combined.view(dtype=[('sx', np.float),
                                             ('sy', np.float),
                                             ('sz', np.float),
                                             ('tx', np.float),
                                             ('ty', np.float),
                                             ('tz', np.float)]))
    n_unique = uniques.size
    unique_tasks = uniques.view(dtype=np.float).reshape(n_unique, 6)
    inds = np.lexsort(unique_tasks.T[-1::-1])
    # Technically the lexsort is not needed since np.unique seems to sort
    # this way anyway, but I prefer to do it explicitly.
    # If speed is ever an issue, this can be revised.
    return unique_tasks[inds]
    
class AlignmentWarning(UserWarning):
    def __init__(self, message):
        UserWarning.__init__(self, message)
    
def _check_times(a0, a1, w0, w1):
    valid = ~(np.isnan(a0) | np.isnan(a1))
    if not np.all(w0[valid] <= a0[valid]):
        warn(AlignmentWarning("trial %d: alignment up time prior to "
                              "window start"))
    if not np.all(a0[valid] < a1[valid]):
        warn(AlignmentWarning("trial %d: alignment up time prior to "
                              "alignment down time"))
    if not np.all(a1[valid] <= w1[valid]):
        warn(AlignmentWarning("trial %d: alignment down time following "
                              "window finish"))
        

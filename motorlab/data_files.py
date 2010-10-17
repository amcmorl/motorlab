import numpy as np
from scipy import io
import os.path
#from tuning_change.util import twopt_interp
import csv

version = [0.2, '10/08/13 -- Angus McMorland']

versions = ['3d_data_2.1', 'VR_RTMA_1.0']

def pdbg(i, *args):
    if i < 3:
        print args
        
class ExpFile:
    file_name = None
    version = None
    velocities = None
    times = None

    def __init__(self, file_name, version):
        '''
        Default efile constructor- needs to be over-ridden by subclasses.
        
        Parameters
        ----------
        file_name : string
          path to file to load
        version : string
          one of 'VR_RTMA_10' | '3d_data_21'
        '''
        self.file_name = file_name
        self.version = version
        #print "Loading file %s..." % (os.path.split(file_name)[1])

################################# VR_RTMA_10 #################################

class CenOut_VR_RTMA_10_File(ExpFile):
    '''
    Handler class for VR RTMA 1.0 formatted data files.
    
    Parameters
    ----------

    Notes
    -----
    This is the current file version.
    '''
    def __init__(self, file_name):
        version = 'VR_RTMA_1.0'
        ExpFile.__init__(self, file_name, version)
        self.load_data()
        
    def load_data(self):
        '''Extracts data from VR_RTMA v1.0 data files
        and converts kinematic data from optotrac to monkey co-ordinates

        '''
        file_dat = io.loadmat(self.file_name, squeeze_me=True,
                              struct_as_record=False)
        self.trials = file_dat['trials']
        self.HoldAStart = self.trials.HoldAStart
        self.HoldAFinish = self.trials.HoldAFinish
        self.HoldBStart = self.trials.HoldBStart
        self.HoldBFinish = self.trials.HoldBFinish
        self.ReactionFinish = self.trials.ReactionFinish
        self.TargetPos = self.trials.TargetPos
        self.StartPos = self.trials.StartPos
        self.PlexonTrialTime = self.trials.PlexonTrialTime
        self.kins = file_dat['kinematics']
        self.spikes = np.asarray(file_dat['spikes'])
        fix_kin(self.kins, file_dat['header'].CursorTransform)

    def sort_spikes(self, unit, lag=0.):
        '''Sort spikes for the given unit into trials.

        Parameters
        ----------
        unit : string
          name of unit to extract
        lag : float
          lag time (of kinematics relative to neural) in s
          positive lag implies neural event precedes kinematic

        Returns
        -------
        spikes_sorted : list of 1-d arrays
          each item is spike times in a given trial,
          relative to PlexonTrialTime for that trial start'''
        spikes_sorted = []
        spike_times = np.asarray(self.spikes.item().__dict__[unit] + lag)
        n_trials = self.trials.PlexonTrialTime.size
        for i in xrange(0, n_trials):
            valid_spikes = np.asarray( \
                (spike_times > self.trials.PlexonTrialTime[i] + \
                     self.trials.HoldAStart[i]) &
                (spike_times < (self.trials.PlexonTrialTime[i] + \
                                    self.trials.HoldBFinish[i])))
            spikes_sorted.append(spike_times[valid_spikes] - 
                                 self.trials.PlexonTrialTime[i])
        return spikes_sorted

    def get_kinematics(self):
        """Returns an array containing all the trial's kinematic data.

        Each entry in time has 4d - x,y,z,t
        where t is relative to spike start.

        Parameters
        ----------
        self : CenOut_VR_RTMA_10_File
        
        Returns
        -------
        kins_list : list of arrays, each with shape (n, 4)
          which are x,y,z,t points for each position stored
        """
        kins = self.kins
        trials = self.trials
        kins_list = []
        n_trials = trials.HoldAStart.size
        start_time = trials.PlexonTrialTime + trials.HoldAStart
        stop_time = trials.PlexonTrialTime + trials.HoldBFinish
        for i in xrange(n_trials):
            # windowed from HoldAStart to HoldBFinish - was from HoldAStart
            valid_idxs = ((kins.PlexonTime > start_time[i]) & \
                         (kins.PlexonTime < stop_time[i]))
            valid_times = kins.PlexonTime[valid_idxs] \
                - trials.PlexonTrialTime[i]
            valid_positions = kins.Markers[valid_idxs]
            valid_kins = np.hstack((valid_positions, valid_times[...,None]))
            kins_list.append(valid_kins)
        return kins_list
    
def fix_kin(kin, tr, threshold=1.):
    '''Wherever first three columns of tr are not 1 (and not 0),
    and have only one entry per row, divide fourth column by non-zero entry
    and set also set non-zero entry to 1.
    Also replaces all values > 1 (unlikely) with nans.'''

    shp = kin.Markers.shape
    m = np.dot(np.hstack((kin.Markers, np.ones((shp[0], 1)))), \
                   tr.transpose())
    bigs = np.abs(m[:,0:3] > 1.)
    m[bigs] = np.nan
    kin.Markers = m[:,0:3]

################################# 3d_data_21 #################################

class CenOut_3d_data_21_File(ExpFile):

    def __init__(self, file_name):
        version = '3d_data_2.1'
        ExpFile.__init__(self, file_name, version)
        self.load_data()

    def load_data(self):
        '''extracts data from 3d_data_2.1 (old) data files
        and converts kinematic data from optotrac to monkey co-ordinates

        doesn''t directly handle old binary format, so requires conversion
        with matlab into .mat file first

        currently no fix_kin done to profiles!!! not sure whether it''s
        necessary, or how to get transformation matrix
        '''
        file_dat = io.loadmat(self.file_name, squeeze_me=True,
                              struct_as_record=False)
        self.trials = file_dat['trials']
        trial_pars = self.get_trial_pars()
        self.HoldAStart, self.HoldAFinish = trial_pars[0:2]
        self.HoldBStart, self.HoldBFinish = trial_pars[2:4]
        self.TargetPos = trial_pars[4]
        self.StartPos = np.zeros_like(self.TargetPos)
        self.head = file_dat['header']

    def sort_spikes(self, unit_name):
        '''Sort spikes for the given unit into trials.

        Returns
        -------
        spikes_sorted : list of 1-d arrays'''
        spikes_sorted = []
        #n_trials = self.trials.size
        for trial in self.trials:
            unit_idx = get_unit_idx(unit_name, trial)
            spikes_sorted.append(np.asarray(trial.Spikes.ts[unit_idx]))
        return spikes_sorted

    def get_trial_pars(self):
        """returns the equivalent of the new file format's HoldAStart"""
        n_trials = self.trials.size
        HoldAStart = np.empty(n_trials) + np.nan
        HoldAFinish = np.empty(n_trials) + np.nan
        HoldBStart = np.empty(n_trials) + np.nan
        HoldBFinish = np.empty(n_trials) + np.nan
        TargetPos = np.empty((n_trials, 3)) + np.nan
        StartPos = np.empty((n_trials, 3)) + np.nan
        for i, trial in enumerate(self.trials):
            HoldAStart[i] = 0 # relative to spike time start!!
            HoldAFinish[i] = trial.HoldATime
            HoldBStart[i] = trial.HoldATime \
                            + trial.DelayTime \
                            + trial.ReactionTime \
                            + trial.MovementTime
            HoldBFinish[i] = trial.HoldATime \
                             + trial.DelayTime \
                             + trial.ReactionTime \
                             + trial.MovementTime \
                             + trial.HoldBTime
            #TargetPos[i] = get_target_pos(trial)
            TargetPos[i] = trial.TargetModelPos / 1000.
        return HoldAStart, HoldAFinish, \
            HoldBStart, HoldBFinish, \
            TargetPos, StartPos

    def get_kinematics(self):
        """Returns an array containing all the trials' kinematic data.
        Problem is each kinematic set has a slightly different length,
        so could return as a list of arrays.
        Each entry in time has 4d - x,y,z,t
        where t is relative to spike start.

        Returns
        -------
        kins_list : list of arrays, each with shape (n, 4)
          which are x,y,z,t points for each position stored
        """
        kins_list = []
        for i, trial in enumerate(self.trials):
            n_bins = trial.Kinematics.shape[1]
            Sync = -trial.SyncTime
            RT = trial.ReactionTime
            DT = trial.DelayTime
            HAT = trial.HoldATime
            t_offset = Sync - RT - DT - HAT # = AII + AST - delta
            kin = np.empty((n_bins, 4)) + np.nan
            samp_f = self.head.OptotrakSamplingFreq
            kin[:, -1] = np.arange(n_bins) * 1000 / samp_f - t_offset
            # subtracting offset makes times relative to spike 0

            # this also performs fix_kin duties, transposing/flipping axes to
            # agree with TargetModelPos
            kin_temp = (trial.Kinematics.T - self.head.CenterPosition[None,...]) \
                / 1000. # divide by 1000 makes the units in line with VR_RTMA format
            kin[:, 0:2] = kin_temp[:,1::-1]
            kin[:, 2] = -kin_temp[:, 2]
            kins_list.append(kin)
        return kins_list

############################## UTILITY ROUTINES ##############################

def get_unit_idx(unit, trial):
    chans = trial.Spikes.chan
    return np.nonzero(chans == unit)[0].item()

def find_common_units(files):
    unit_set = set(list_units(files[0]))
    for f in files[1:]:
        these_units = list_units(f)
        unit_set = unit_set.intersection( set(these_units) )
    return list(unit_set)

class units_file_dialect(csv.Dialect):
    delimiter = ' '
    skipinitialspace = True
    quoting = csv.QUOTE_NONE
    lineterminator = '\n'

def compile_units_list(units_file):
    f = open(units_file, 'r')
    units_list = []
    reader = csv.reader(f, dialect=units_file_dialect)
    for row in reader:
        date = row[0]
        files_first, files_last = [int(x) for x in row[1:3]]
        units = row[3:]
        for unit in units:
            unit_listing = [date, unit, files_first, files_last]
            units_list.append(unit_listing)
    return units_list

def convert_to_long_unit_name(short_unit_name, M1=True):
    if M1: M1 = 100
    else: M1 = 0
    chan_num = M1 + int(short_unit_name[:-1])
    unit_num = ord(short_unit_name[-1]) - 64
    return 'Unit%03d_%d' % (chan_num, unit_num)

def list_units(fname, version='VR_RTMA_1.0'):
    assert (version in versions) or (version in range(len(versions)))
    assert type(fname) == str

    nfname = os.path.normpath(fname)
    if not os.path.exists(nfname):
        raise ValueError, fname + " does not exist."

    if type(version) == int:
        version = versions[version]

    f = io.loadmat(nfname, squeeze_me=True, struct_as_record=False)
    if version == 'VR_RTMA_1.0':
        units = f['spikes']._fieldnames
    elif version == '3d_data_2.1':
        units = f['trials'][0].Spikes.chan
    return list(units)

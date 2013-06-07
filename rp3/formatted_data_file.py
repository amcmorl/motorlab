import numpy as np
from vecgeom import norm
import os.path
import h5py
import scipy.io as io
import types
import motorlab.rp3.rotations as rt

'''
Utility functions for handling formatted data (from the RP3 experiment).
'''
# codes assigned by executive for each task phase
TE_HAND_OPENING = 20
TE_REACHING = 6
TE_HOMING = 7

get_data = lambda d,v,t: np.asarray(d[v], dtype=t).squeeze()

def get_mat_field(self, *field):
    '''
    field loader for matlab pre v7.3 format files that accepts same
    signature as hdf5_field_loader

    Parameters
    ----------
    f : dict
      loaded matlab file
    field : list of string
      data tree path to field
    '''
    field = list(field)
    current = self.f[field.pop(0)]
    for layer in field:
        current = current.__dict__[layer]
    return current

def get_hdf5_field(self, *field):
    '''
    Parameters
    ----------
    f : h5py.highlevel.File
      loaded hdf5 file
    field : list of string
      data tree path to field
    '''
    fieldname = '/'.join(field)
    return self.f[fieldname]

class FormattedData(object):
    '''
    generic formatted data file wrapper

    Attributes
    ----------
    fname : string
      associated file name
    format : string
      either 'hdf5' or 'mat'

    Methods
    -------
    get_field -

    Parameters
    ----------
    fname : string
      path to file
    '''
    def __init__(self, fname):
        raise DeprecationWarning("Use motorlab.dfile instead.")
        if not os.path.exists(fname):
            raise ValueError('%s does not exist' % (fname))
        else:
            self.fname = fname
        if h5py.is_hdf5(fname):
            self.format = 'hdf5'
            self.f = h5py.File(fname)
            self.get_field = types.MethodType(get_hdf5_field, self)
        else:
            self.format = 'mat'
            self.f = io.loadmat(fname, struct_as_record=False,
                                squeeze_me=True)
            self.get_field = types.MethodType(get_mat_field, self)

#======================================================================

def get_pts(fData, trials=None, phases=None):
    '''
    get boolean vector index into time-series data corresponding to
    the given trials and phases
    '''
    trial_no = get_data(fData, 'TrialNo', int)
    if trials != None:
        trial_pts = np.in1d(trial_no, trials)
    else:
        # all true by default
        trial_pts = np.ones_like(trial_no, dtype=bool)
    task_state_codes = get_data(fData, 'TaskStateCodes/Values', int)
    if phases != None:
        phase_pts = np.in1d(task_state_codes, phases)
    else:
        # all true by default
        phase_pts = np.ones_like(task_state_codes, dtype=bool)
    return trial_pts & phase_pts

def get_translation_dist_to_targ(fData, pt=None, planner=False):
    '''
    '''
    if pt == None:
        pt = slice(None)

    if planner:
        target_field = 'Position/PlannerTarget'
    else:
        target_field = 'Position/target'
    targ_pos = np.asarray(fData[target_field])[pt,0:3]
    curr_pos = np.asarray(fData['Position/Actual'])[pt,0:3]
    dist = norm(targ_pos - curr_pos, axis=1)
    return dist

def get_orientation_dist_to_targ(fData, pt=None, planner=False):
    '''
    '''
    if pt == None:
        pt = slice(None)

    if planner:
        target_field = 'Position/PlannerTarget'
    else:
        target_field = 'Position/target'
    targ_cori = rt.eul2DCM(fData[target_field][pt,0:3], 'xyz')
    curr_cori = fData['Position/Actual'][pt,0:3]
    dist = rt.DCMdiff(targ_cori, curr_cori)
    return dist




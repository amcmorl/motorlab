import numpy as np
import proc_tools
from vecgeom import norm, unitvec

def get_vel(pos, time, tax=0, spax=-1):
    ''' Get instantaneous velocity

    Parameters
    ----------
    time : array_like

    pos : array_like

    tax : int, optional
      time axis, defaults to 0
      has to be suitable for both pos and time,
      so probably needs to be +ve, i.e. indexed from beginning
    spax : int, optional
      space axis in pos, defaults to -1, i.e. last axis
    '''
    dp = np.diff(pos, axis=tax)
    dt = np.diff(time, axis=tax)

    if np.rank(dp) != np.rank(dt):
        if spax < 0:
            spax = len(pos.shape) + spax
        dts = [slice(None) for x in dt.shape]
        dts.insert(spax, None)
    else:
        dts = [slice(None) for x in dt.shape]
    return dp / dt[dts]

def get_dir(pos, tax=-2, spax=-1):
    '''Get instantaneous direction

    Parameters
    ----------
    pos : array_like

    tax : int, optional
      time axis, defaults to 0
    '''    
    dp = np.diff(pos, axis=tax)
    return unitvec(dp, axis=spax)

def get_speed(pos, time, tax=0, spax=-1):
    '''Get speed

    Parameters
    ----------
    pos : array_like
      
    time : array_like
    
    tax : int, optional
      time axis, defaults to 0
    spax : int, optional
      space axis, defaults to -1, i.e. last axis
    '''    
    vel = get_vel(pos, time, tax=tax)
    return norm(vel, axis=spax)

def get_tgt_dir(pos, tgt):
    '''
    Calculate direction from current position to target.

    Parameters
    ----------
    pos : ndarray
      positional data,
      shape (ntask, nbin, ndim) or (ntask, nrep, nbin, ndim)
    tgt : ndarray
      target positions, shape (ntask, ndim)

    Notes
    -----
    Not yet made suitable for tax and spax variants.
    '''
    if np.rank(pos) == 3: # no repeats
        tgt = tgt[:,None]
    elif np.rank(pos) == 4: # with repeats
        tgt = tgt[:,None,None]
    drn = tgt - pos
    drn /= norm(drn, axis=-1)[...,None]
    return drn

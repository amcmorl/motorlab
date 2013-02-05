import numpy as np
from amcmorl_py_tools.vecgeom import norm
import amcmorl_py_tools.vecgeom.stats as ss
from scipy.stats.distributions import poisson
from motorlab.kinematics import get_tgt_dir, get_dir
from amcmorl_py_tools.curves import squash, set_squash_defaults
import ConfigParser
from amcmorl_py_tools import curves

#==============================================================================
# kinematics
#==============================================================================

def read_task(file, section, set_name):
    '''
    Returns
    -------
    tasks : ndarray
      formatted as (ntask, ndim * 2)
      where the 2nd dimension is start position, finish position
    '''
    config = ConfigParser.ConfigParser()
    config.read(file)
    lines = config.get(section, set_name).strip().split('\n')
    return np.array([[float(y) for y in x.split()] for x in lines])

def make_simple_trajectories(tasks, npt=201,
                             speed_mid=0.5, speed_width=0.2):
    '''
    Make linear trajectories from start to finish
    with a gaussian speed profile.

    Parameters
    ----------
    tasks : ndarray
      shape (n, 6)
      n tasks with start position :3 and target position 3:
    '''
    ndim = tasks.shape[1] / 2
    time = np.linspace(0, 1, npt)
    spd = curves.gauss1d(time, 1., speed_width, speed_mid)
    spd /= np.sum(spd)
    start, stop = tasks[:,:ndim], tasks[:,ndim:]
    drn = stop - start
    vel = spd[None,:,None] * drn[:,None,:] # shape (task, time, space)
    disp = np.cumsum(vel/2. * time[None,:,None], axis=1) # kinematic eqn
    pos = disp + start[:,None]
    return time, pos, spd

#==============================================================================
# rates
#==============================================================================

def make_rate(time, pos, model, **kwargs):
    '''
    Parameters
    ----------
    pos : ndarray
      positional data
    model : str
      code for model to construct
    
    Other Parameters
    ----------------
    bs : float
    b0 : float
    PD : array_like
     3-vector
    kd : float
    PP : array_like
      3-vector
    kp : float
    '''
    pass

def make_inst_dir_rate(time, pos, pd, kd, sqp={}):
    '''
    Create directional firing components from instantaneous direction,
    preferred direction, and modulation depth.

    Parameters
    ----------
    time : ndarray
      times for pos samples, shape (??)
    pos : ndarray
      positional data, shape (ntrial, nbin, 3)
    pd : array_like
      pds for a number of cells, shape (nunit, 3)
    kd : array_like
      modulation depth, shape (nunit)
    sqp : dict, optional
      parameter values for squashing function
    '''
    drn = get_dir(pos, tax=1, spax=-1)
    rate = np.dot(drn, pd.T) * kd
    set_squash_defaults(sqp, rate)
    return squash(rate, sqp)

def make_tgt_dir_rate(tgt, pos, pd, kd, sqp={}):
    '''
    Create directional firing components from target (target - current)
    direction, and modulation depth.

    Parameters
    ----------
    task : ndarray
      start and target positions for each task, shape (ntask, 6)
    pos : ndarray
      cursor position, shape (ntask, nrep, nbin, ndim)
    pd : ndarray
      preferred directions, shape (nunit, 3)
    kd : ndarray
      modulation depth, shape (nunit)
    '''
    drn = get_tgt_dir(pos, tgt)
    rate = np.dot(drn, pd.T) * kd
    set_squash_defaults(sqp, rate)
    return squash(rate, sqp)
       
def generate_firing_rate2(pos, PD=None, PG=None,
                          k={}, tau={}, model='kd'):
    ''' Generate simulated firing rate of a single cell during a single reach.

    Parameters
    ----------
    pos : array_like, shape (n_bins, 3)
        x,y,z position
    PD : array_like, shape (3,) OR shape (n_bins - 1, 3) for changing PD
        prefered direction
    PG : array_like, shape (3,)
        positional gradient
    k : dictionary
        scale factors for components of firing
    tau : dictionary
        lags for components of firing
    model : string
        see module docstring for details
        
    Returns
    -------
    fr : masked_array_like, shape (n_bins - 1)
        firing rates
    PD : array, shape (3,) or (n_bins, 3)
        preferred direction(s) used for model
        useful for comparing with recovered PDs from simulated data

    Notes
    -----
    potential models of lag:
    1) all terms have same lead/lag
    2) position and velocity have different lags (then what about speed?)
    3) all terms have a different lag
    '''    
    assert np.rank(pos) == 2
    assert pos.shape[1] == 3
    n_bins = pos.shape[0]
    if np.rank(PD) == 1:
        assert (len(PD) == 3)
    else:
        assert PD.shape[1] == 3
    assert len(PG) == 3
    assert type(k) == type(tau) == dict
    assert type(model) == str

    k_defaults = {'k' : 1., 'p' : 1.,
                  'd' : 1., 'v' : 1.,
                  's' : 1., 'n' : 100,
                  'F' : 1.}
    for key, val in k_defaults.iteritems():
        k.setdefault(key, val)
    tau_defaults = {'p' : 0, 'd' : 0, 'v' : 0, 's' : 0}
    for key, val in tau_defaults.iteritems():
        tau.setdefault(key, val)        

    # calculate directions
    deltas_inst = np.empty_like(pos)
    
    # since diff calculate trailing edges of bins
    deltas_inst[1:,...] = np.diff(pos, axis=0)
    deltas_inst[0,:] = deltas_inst[1,...]
    speed = np.sqrt(np.sum(deltas_inst**2, axis=1))
    drn = deltas_inst / speed[...,None]
    speed /= speed.max()
    #assert np.all(np.abs(lens_inst/lens_inst.max() - speed/speed.max()) < 1e-6)
    
    fr = np.ma.zeros(n_bins)

    if 'F' in model:
        # Fisher-function distribution
        # find angle between PD and d
        if np.rank(PD) == 1:
            theta = np.arccos(np.dot(drn, PD))
        else:
            raise NotImplementedError, 'Sorry Fisher tuning and changing' \
                ' PDs has yet to be implemented'
        # calculate h(theta, kappa) (kappa = k['F'])
        h = ss.vmf_pde_centered(theta, k['F'])
    
    if 'k' in model:
        # constant term
        fr += k['k']

    if 'p' in model:
        # positional gradient
        pos_component = k['p'] * np.sum(PG[None,...] * pos[:,...], axis=1)
        #... pos.shape = (n_bins, 3)
        fr = add_with_lag(fr, pos_component, tau['p'])

    if 'd' in model:
        if 'F' in model: # Fisher tuning
            d_component = k['d'] * h
            #does this scale okay? - from 0 at great distance to..? I think so.
        else: # cosine tuning
            # directional component
            if np.rank(PD) == 1:
                # have static PD
                d_component = k['d'] * np.sum(PD[None,...] * drn, axis=1)
            else:
                # have changing PD
                d_component = k['d'] * np.sum(PD * drn, axis=1)
            # PD[...,None].shape == (3,1), drn.shape == (3,199),
        fr = add_with_lag(fr, d_component, tau['d'])

    if 'v' in model:
        # gain-field modulated speed (i.e. speed dot (direction from PD))
        if 'F' in model: # Fisher tuning
            v_comp = k['v'] * speed * h
        else: # cosine tuning
            if np.rank(PD) == 1:
                v_comp = k['v'] * speed * np.sum(PD[None,...] * drn, axis=1)
            else:
                v_comp = k['v'] * speed * np.sum(PD * drn, axis=1)
        fr = add_with_lag(fr, v_comp, tau['v'])

    if 's' in model:
        # additive speed component
        s_comp = k['s'] * speed
        fr = add_with_lag(fr, s_comp, tau['s'])
        fr += k['s'] * speed

    if 'n' in model:
        # noise
        fr[fr < 0] = 1e-8
        mask = fr.mask.copy()
        fr[:] = poisson.rvs(fr * k['n'], size=fr.shape) / float(k['n'])
        fr.mask |= mask
    return fr

def add_with_lag(fr, component, lag):
    assert np.rank(fr) == 1
    if lag == 0:
        fr += component
    else:
        fr += np.roll(component, lag)
        #... select only valid region
        low = lag if lag > 0 else None
        high = lag if lag < 0 else None
        mask = np.ones(fr.shape[0], dtype=bool)
        mask[low:high] = False
        fr.mask |= mask
    return fr

def generate_changing_PD(PDa, PDb, n_pts):
    '''Returns an array (time.size, 3) of PDs that smoothly and linearly changes from PDa to PDb in `time.size` bins.

    Parameters
    ----------
    PDa : sequence (3,)
    PDb : sequence (3,)
        starting and finishing PDs of result
    n_pts : integer
        number of time points to interpolate

    Returns
    -------
    PDs : ndarray (t, 3)
        smoothly, linearly changing PDs from PDa to PDb
    '''
    PDa = np.asarray(PDa)
    PDb = np.asarray(PDb)
    assert PDa.size == PDb.size == 3
    assert np.allclose(norm(PDa), 1)
    assert np.allclose(norm(PDb), 1)
    angle_between = np.arccos(np.dot(PDa, PDb))
    angles = np.linspace(0, angle_between, n_pts)
    origin = np.zeros(3)
    PD = np.array([ss.parameterized_circle_3d(angle, PDa, PDb, origin)
                   for angle in angles])
    return PD


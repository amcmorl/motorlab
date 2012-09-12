import numpy as np
import curves
import matplotlib.pyplot as plt
import myutils
from warnings import warn
from scipy.optimize import fmin
from curves import gauss1d
import motorlab.tuning.simulate.rates as sim_rates

print "tuning_change.firing_models is deprecated.\n"
"Use tuning_change.simulate instead."

'''
Generates firing rates associated with simulated movements.

The model code is as follows. Single letter codes for components:
    k = offset
    p = position
    d = direction
    v = velocity,
    s = speed,
    n = noise (Poisson)
    N = slow timescale noise applied as baseline shift to each trial          
and the following modifiers:
    F = Fisher directional tuning curve (k['F'] is kappa, spread parameter)
    X = changing PD

Set each trial to take 1 normalized time interval to complete.

Will model movement as having a gaussian speed profile, starting
and ending at 0. Max speed will be normalized to 1.

Acceleration will be the derivative of speed,
and displacement its integral.

Positions, directions and velocities are calculated for each of the targets.

Curved movements are modelled so that the velocity vector in the direction perpendicular to the target direction follows a sine curve, and the parallel velocity vector modified to be the vector difference between the straight line movement's velocity vector (derived from a gaussian) and the perpendicular vector (derived from the sine wave).

To Do
-----
* add \'affine\' axes / displacement into model
* investigate how PD changes could get larger /
  what limiting factors are for PD changes when position is taken into account
* some interaction between position and direction
  - i.e. convergence of PG and PD??
'''

rt2 = np.sqrt(2)

targets_dict = \
    {'co_2d_8' : np.asarray(\
        [[0.,  0.,  0.,  rt2,   0., 0.],
         [0.,  0.,  0.,   1.,  -1., 0.],
         [0.,  0.,  0.,   0., -rt2, 0.],
         [0.,  0.,  0.,  -1.,  -1., 0.],
         [0.,  0.,  0., -rt2,   0., 0.],
         [0.,  0.,  0.,  -1.,   1., 0.],
         [0.,  0.,  0.,   0.,  rt2, 0.],
         [0.,  0.,  0.,   1.,   1., 0.]]),
                
     'co_oc_2d_8' : np.asarray(\
        [[  0.,   0.,  0.,  rt2,   0., 0.],
         [ rt2,   0.,  0.,   0.,   0., 0.],
         [  0.,   0.,  0.,   1.,  -1., 0.],
         [  1.,  -1.,  0.,   0.,   0., 0.],
         [  0.,   0.,  0.,   0., -rt2, 0.],
         [  0., -rt2,  0.,   0.,   0., 0.],
         [  0.,   0.,  0.,  -1.,  -1., 0.],
         [ -1.,  -1.,  0.,   0.,   0., 0.],
         [  0.,   0.,  0., -rt2,   0., 0.],
         [-rt2,   0.,  0.,   0.,   0., 0.],
         [  0.,   0.,  0.,  -1.,   1., 0.],
         [ -1.,   1.,  0.,   0.,   0., 0.],
         [  0.,   0.,  0.,   0.,  rt2, 0.],
         [  0.,  rt2,  0.,   0.,   0., 0.],
         [  0.,   0.,  0.,   1.,   1., 0.],
         [  1.,   1.,  0.,   0.,   0., 0.]]),

     'co_3d_8' : np.asarray(\
        [[0.,  0.,  0., -1,  -1., -1.],
         [0.,  0.,  0., -1., -1.,  1.],
         [0.,  0.,  0., -1.,  1., -1.],
         [0.,  0.,  0., -1.,  1.,  1.],
         [0.,  0.,  0.,  1., -1., -1.],
         [0.,  0.,  0.,  1., -1.,  1.],
         [0.,  0.,  0.,  1.,  1., -1.],
         [0.,  0.,  0.,  1.,  1.,  1.]]),
                
     'co_oc_3d_8' : np.asarray( \
        [[ 0.,  0.,  0., -1., -1., -1.],
         [-1., -1., -1.,  0.,  0.,  0.],
         [ 0.,  0.,  0., -1., -1.,  1.],
         [-1., -1.,  1.,  0.,  0.,  0.],
         [ 0.,  0.,  0., -1.,  1., -1.],
         [-1.,  1., -1.,  0.,  0.,  0.],
         [ 0.,  0.,  0., -1.,  1.,  1.],
         [-1.,  1.,  1.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  1., -1., -1.],
         [ 1., -1., -1.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  1., -1.,  1.],
         [ 1., -1.,  1.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  1.,  1., -1.],
         [ 1.,  1., -1.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  1.,  1.,  1.],
         [ 1.,  1.,  1.,  0.,  0.,  0.]]),

     'co_oc_3d_26' : np.asarray(\
        [[ 0.     ,  0.     ,  0.     , -1.     , -1.     , -1.     ],
         [-1.     , -1.     , -1.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474, -1.22474,  0.     ],
         [-1.22474, -1.22474,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.     , -1.     ,  1.     ],
         [-1.     , -1.     ,  1.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474,  0.     , -1.22474],
         [-1.22474,  0.     , -1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.73205,  0.     ,  0.     ],
         [-1.73205,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474,  0.     ,  1.22474],
         [-1.22474,  0.     ,  1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.     ,  1.     , -1.     ],
         [-1.     ,  1.     , -1.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474,  1.22474,  0.     ],
         [-1.22474,  1.22474,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.     ,  1.     ,  1.     ],
         [-1.     ,  1.     ,  1.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     , -1.22474, -1.22474],
         [ 0.     , -1.22474, -1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     , -1.73205,  0.     ],
         [ 0.     , -1.73205,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     , -1.22474,  1.22474],
         [ 0.     , -1.22474,  1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     , -1.73205],
         [ 0.     ,  0.     , -1.73205,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.73205],
         [ 0.     ,  0.     ,  1.73205,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  1.22474, -1.22474],
         [ 0.     ,  1.22474, -1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  1.73205,  0.     ],
         [ 0.     ,  1.73205,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  1.22474,  1.22474],
         [ 0.     ,  1.22474,  1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.     , -1.     , -1.     ],
         [ 1.     , -1.     , -1.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474, -1.22474,  0.     ],
         [ 1.22474, -1.22474,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.     , -1.     ,  1.     ],
         [ 1.     , -1.     ,  1.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474,  0.     , -1.22474],
         [ 1.22474,  0.     , -1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.73205,  0.     ,  0.     ],
         [ 1.73205,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474,  0.     ,  1.22474],
         [ 1.22474,  0.     ,  1.22474,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.     ,  1.     , -1.     ],
         [ 1.     ,  1.     , -1.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474,  1.22474,  0.     ],
         [ 1.22474,  1.22474,  0.     ,  0.     ,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.     ,  1.     ,  1.     ],
         [ 1.     ,  1.     ,  1.     ,  0.     ,  0.     ,  0.     ]]),
     
     'co_3d_26' : np.asarray(\
        [[ 0.     ,  0.     ,  0.     , -1.     , -1.     , -1.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474, -1.22474,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.     , -1.     ,  1.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474,  0.     , -1.22474],
         [ 0.     ,  0.     ,  0.     , -1.73205,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474,  0.     ,  1.22474],
         [ 0.     ,  0.     ,  0.     , -1.     ,  1.     , -1.     ],
         [ 0.     ,  0.     ,  0.     , -1.22474,  1.22474,  0.     ],
         [ 0.     ,  0.     ,  0.     , -1.     ,  1.     ,  1.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     , -1.22474, -1.22474],
         [ 0.     ,  0.     ,  0.     ,  0.     , -1.73205,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     , -1.22474,  1.22474],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     , -1.73205],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.73205],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  1.22474, -1.22474],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  1.73205,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  0.     ,  1.22474,  1.22474],
         [ 0.     ,  0.     ,  0.     ,  1.     , -1.     , -1.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474, -1.22474,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.     , -1.     ,  1.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474,  0.     , -1.22474],
         [ 0.     ,  0.     ,  0.     ,  1.73205,  0.     ,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474,  0.     ,  1.22474],
         [ 0.     ,  0.     ,  0.     ,  1.     ,  1.     , -1.     ],
         [ 0.     ,  0.     ,  0.     ,  1.22474,  1.22474,  0.     ],
         [ 0.     ,  0.     ,  0.     ,  1.     ,  1.     ,  1.     ]])}
    
def generate_ballistic_profiles(fwhm=0.2, center=0.5, n_pts=200):
    '''
    Generates base values for a ballistic movement.

    Parameters
    ----------
    fwhm : float
      curve parameter for speed profile
    center : float
      position of peak speed in profile
    n_pts : int, default=200
      number of points to generate
    
    Returns
    -------
    time : ndarray, shape (n_pts,)
      time co-ordinates
    disp : ndarray, shape (n_pts,)
      displacement through movement
    speed : ndarray, shape (n_pts,)
      speed during movement
    acc : ndarray, shape (n_pts - 1,)
      acceleration during movement

    Notes
    -----    
    '''
    assert type(0.) == type(fwhm) == type(center)
    assert type(0) == type(n_pts)
    time = np.linspace(0, 1., n_pts, endpoint=True)
    speed = curves.gauss1d(time, 1., fwhm, center)
    disp = np.cumsum(speed)
    disp /= disp.max()
    acc = np.diff(speed)
    warn("acc should be calculated using a divide by time")
    return time, disp, speed, acc

def normalize_direction(dk):
    '''Normalize a triplet into a direction.

    Parameters
    ----------
    dk : ndarray (3,)
      x,y,z components of a direction

    Returns
    -------
    d : ndarray (3,)
      unit length vector, same direction as dk
    '''
    assert type(dk) == np.ndarray
    k = np.sqrt(np.sum(dk ** 2))
    if k <> 0:
        return dk / k
    else:
        return np.zeros(3)

def generate_cell_property(which='PD'):
    '''Generate the preferred direction (c coefficients) for a cell.

    Parameters
    ----------
    PDk : array_like, shape (3,)
        3d coefficients of firing rate defining the directional tuning
    PGk : array_like, shape (3,)
        3d coefficients of firing rate defining the positional tuning
    '''
    assert type(which) == str

    if which == 'PD':
        dk = np.asarray([1.0, 0.3, 0.2])
    elif which == 'PG':
        dk = np.asarray([1., -0.8, -0.1])
    elif which == 'PDb':
        dk = np.asarray([-0.8, 0.1, -0.7])
    elif which == 'random':
        dk = np.random.random(3)
    return normalize_direction(dk)

def calc_pos_and_dir_along_movement(disp, start, stop):
    '''Calculate the positions of the cursor at
    each time point, for a movement from `tasks[:,:3]` to `tasks[:,3:]`

    Parameters
    ----------
    disp : array_like, shape (n_pts,)
        displacement through task
    start : array_like
        x,y,z triplet of co-ordinates of starting point
    stop : array_like
        x,y,z triplet of co-ordinates of stopping point

    Returns
    -------
    pos : ndarray, shape (n_pts, 3)
        x,y,z position during task
    drn : ndarray, shape (n_pts - 1, 3)
        instantaneous x,y,z direction during task
        
    Notes
    -----
    This algorithm currently makes the assumption all movements of interest
    are the same distance, since they all take the same time.
    '''
    assert np.rank(disp) == 1
    assert len(stop) == len(start) == 3

    vec = stop - start
    #print np.sqrt(np.sum(vec**2))
    pos = vec * disp[...,np.newaxis] + start
    #pos = pos.T
    drn = np.diff(pos, axis=0) # gets direction vector
    drn /= np.sqrt(np.sum(drn**2, axis=1))[...,None] # normalize vector lengths
    return pos, drn

def generate_speed_profile(fwhm=0.2, center=0.5, n_bins=200):
    '''Generates base values for a ballistic movement.

    Parameters
    ----------
    fwhm : float
      curve parameter for speed profile
    center : float
      position of peak speed in profile
    n_pts : int, default=200
      number of points to generate
    
    Returns
    -------
    time : ndarray, shape (n_pts,)
      time co-ordinates
    disp : ndarray, shape (n_pts,)
      displacement through movement
    speed : ndarray, shape (n_pts,)
      speed during movement
    acc : ndarray, shape (n_pts - 1,)
      acceleration during movement

    Notes
    -----    
    '''
    assert type(fwhm) == type(center) == float
    assert type(n_bins) == int

    #n_pts = n_bins + 1
    points = np.linspace(0, 1., n_bins, endpoint=True)
    speed = gauss1d(points, 1., fwhm, center)
    
    return speed#, speed_mid

def opt_above((k, t), sp, c, x, plot=False):
    e, ln = np.exp, np.log
    #t = m * (u - ln(k) / c)
    envelope = ((1 + k) / (1 + e(c * (x - t))) - 1.)
    if plot:
        plt.plot(envelope)
    return sp * envelope

def opt_below((k, t), sp, c, x, plot=False):
    e, ln = np.exp, np.log
    #t = m * (u - ln(1 / k) / c)
    envelope = ((1 + k) / (1 + e(c * (x - t))) - k)
    if plot:
        plt.plot(envelope)
    return sp * envelope

def opt((k, t), above, sp, c, x):
    if above:
        fn = opt_above
    else:
        fn = opt_below
    prof = fn((k, t), sp, c, x)
    negs = prof < 0
    neg_sum = -np.sum(negs * prof)
    pos_sum = np.sum(~negs * prof)
    return (neg_sum - pos_sum)**2

def create_even_perp(speed, curvature=1.2, speed_center=0.5, x_range=2):
    n_bins = speed.size
    x = np.linspace(-x_range, x_range, n_bins)
    x_int = (speed_center - 0.5) * x_range

    # calculate iteration 0
    h0, i0 = 2, 1
    t0 = x_range * (x_int - np.log((h0 - i0) / i0) / curvature)
    envelope = h0 / (1 + np.exp((x - t0) * curvature)) - i0
    
    it0 = speed * envelope
    if it0.sum() > 0.:
        # vary upper asymptote
        (k_f, t_f) = fmin(opt, (.5, t0),
                          args=(True, speed, curvature, x),
                          disp=0, xtol=1e-10)
        even_perp = opt_above((k_f, t_f), speed, curvature, x, plot=False)
    else:
        # vary lower asymptote
        (k_f, t_f) = fmin(opt,(.5, t0),
                          args=(False, speed, curvature, x),
                          disp=0, xtol=1e-10)
        even_perp = opt_below((k_f, t_f), speed, curvature, x, plot=False)
    return even_perp

def calc_pos_along_movement(speed, tasks, curvature=1.2, speed_center=0.5):
    '''Calculate the positions of the cursor at
    each time point, for a movement from `tasks[:,:3]` to `tasks[:,3:]`

    Parameters
    ----------
    disp : array_like, shape (n_pts,)
      displacement through task
    tasks : array_like
      each row is x,y,z for starting point, x,y,z for endpoint
    curvature : float
      amount of curvature of movement (0 < curvature)
      
    Returns
    -------
    pos : ndarray, shape (n_pts, 3)

   x,y,z position during task
        
    Notes
    -----
    This algorithm currently makes the assumption all movements of interest
    are the same distance, since they all take the same time.
    '''
    assert tasks.shape[1] % 2 == 0
    n_dims = tasks.shape[1] / 2.
    assert ((n_dims == 2) or (n_dims == 3))

    #n_bins = speed.size
    if curvature > 0:
        speed_perp = create_even_perp(speed, curvature, speed_center=speed_center)
        assert np.allclose(np.sum(speed_perp), 0.)
    else:
        speed_perp = np.zeros_like(speed)
    speed_para = np.sqrt(speed**2 - speed_perp**2)
    scale_factor = speed_para.sum()
    
    starts, stops = tasks[:,:n_dims], tasks[:,n_dims:]
    para_dirs = stops - starts
    #para_lens = np.sqrt(np.sum(para_dirs**2, axis=1))
    para_vecs = para_dirs[:,None,:] * speed_para[None,:,None] / scale_factor

    if curvature > 0:
        if n_dims == 2:
            y, x = para_dirs.T.copy()
            x[x != 0] *= -1
            y[x == 0] *= -1
            perp_dirs = np.vstack((x,y)).T
            signs = -np.sign(perp_dirs[:,1])[...,None]
        elif n_dims == 3:
            # first switch x & y axes
            y,x,z = para_dirs.T.copy()
            # then multiply either (x | y) by -1
            x[x != 0] *= -1
            y[x == 0] *= -1
            z[:] = 0.
            normals = np.vstack((x,y,z)).T
            perp_dirs = np.cross(normals, para_dirs)
            signs = np.sign(para_dirs[:,2])[...,None]
        signs[signs == 0.] = +1
        perp_dirs *= signs
        perp_lens = np.sqrt(np.sum(perp_dirs**2, axis=1))
        perp_dirs /= perp_lens[...,None]

        # because above doesn't work for vertical trajectories (in 3d)
        perp_dirs[np.isnan(perp_dirs)] = 0.
        assert(~np.any(np.isnan(perp_dirs)))
        perp_vecs = perp_dirs[:,None,:] * speed_perp[None,:,None] / scale_factor

        pos = starts[:,None,:] + np.cumsum(para_vecs, axis=1) + np.cumsum(perp_vecs, axis=1)
    else:
        pos = starts[:,None,:] + np.cumsum(para_vecs, axis=1)
    assert ~np.any(np.isnan(pos))
    return pos

def generate_task_kinematics(tasks=None, mv={}, n_bins=100):
    '''Creates simulated positional data for a set of tasks.
    '''
    assert tasks.shape[1] % 2 == 0

    mv_defaults = {'curvature' : 1.,
                   'center'    : 0.5,
                   'fwhm'      : 0.2}
    mv = myutils.dparse(mv_defaults, mv)

    speed = generate_speed_profile(n_bins=n_bins, fwhm=mv['fwhm'],
                                   center=mv['center'])
    pos = calc_pos_along_movement(speed, tasks, mv['curvature'], mv['center'])
    return pos
    
def do_tasks2(tasks=None, model='kds',
              mv={}, k={}, tau={},
              PD=None, PG=None, PDb=None,
              n_bins=200, full_output=True):
    '''Creates firing rates for each direction requested.

    Parameters
    ----------
    tasks : array_like (n_tasks, 2 * n_dims)
      array of start positions and targets,
      n lots of [x_s, y_s, z_s, x_t, y_t, z_t]
    model : string
      code for model to use, see module docstring for description
    mv : dictionary
      optional movement parameters
      curvature, float, default 1.0
      center, float, default 0.5
      fwhm, float, default 0.2
    k : dictionary
      coefficients of model term coefficients
    tau : dictionary
      lags of firing rate components
    PD : sequence, shape (3,) or None, default None
      preferred direction of the cell
    PG : sequence, shape (3,) or None, default None
      positional gradient of the cell
    PDb : sequence, shape (3,) or None, default None
      final preferred direction, used for changing PD
    n_bins : int, default 200
      number of time points to model in each movement
    full_output : bool, default True
      display lots of output during computation

    Returns
    -------
    speed : ndarray, shape = (n_pts,)
    dirnames : list
      string codes for each direction, in order found in frs
    frs, masked array, shape = (n_dirs, n_pts - 1)
      firing rates in each of the directions
    all_pos : ndarray, shape = (n_dirs, n_pts, 3)
      positions during movements in each of the directions
    '''
    assert type(tasks) == np.ndarray
    assert type(model) == str
    assert type(k) == type(tau) == dict
    assert tasks.shape[1] % 2 == 0
    
    mv_defaults = {'curvature' : 1.,
                   'center'    : 0.5,
                   'fwhm'      : 0.2}
    mv = myutils.dparse(mv_defaults, mv)
    k_defaults = {'k' : 1., 'p' : 1.,
                  'd' : 1., 'v' : 1.,
                  's' : 1., 'n' : 100,
                  'F' : 1.}
    k = myutils.dparse(k_defaults, k)
    tau_defaults = {'p' : 0}
    tau = myutils.dparse(tau_defaults, tau)
    
    # global properties
    speed = generate_speed_profile(n_bins=n_bins, fwhm=mv['fwhm'],
                                   center=mv['center'])
    n_dirs = tasks.shape[0]
    #n_dims = tasks.shape[1] / 2

    # preferred direction
    if PD == None:
        PD = generate_cell_property('PD')
    if PG == None:
        PG = generate_cell_property('PG')
    if ('X' in model):
        if PDb == None:
            PDb = generate_cell_property('PDb')
        PD = sim_rates.generate_changing_PD(PD, PDb, n_bins)
        pds = PD.repeat(n_dirs).reshape(n_bins, 3, n_dirs).transpose([2,0,1])
    else:
        pds = PD.repeat(n_dirs * n_bins).reshape(3, n_dirs, n_bins).transpose([1,2,0])

    # initialize storage variables
    frs = np.ma.empty((n_dirs, n_bins))
    #all_pos = np.empty((n_dirs, n_bins, 3)) + np.nan
    starts = np.tile(tasks[:,:3], n_bins).reshape(n_dirs, n_bins, 3)
    stops = np.tile(tasks[:,3:], n_bins).reshape(n_dirs, n_bins, 3)
    
    pos = calc_pos_along_movement(speed, tasks, mv['curvature'], mv['center'])
    #pos_mids = calc_pos_along_movement(speed_mids, tasks,
    #                                   mv['curvature'], mv['center'])
    
    # loop over targets
    for i_dir, task in enumerate(tasks):
	if ('N' in model) and ('k' in model):
            k['k'] *= np.random.normal(loc=1., scale=k['N'])

        frs[i_dir] = sim_rates.generate_firing_rate2(pos[i_dir],
                                           #dirs_inst[i_dir], speed,
                                           PD, PG,
                                           k=k, tau=tau, model=model)    
    if full_output:
        return speed, frs, pos, pds, starts, stops
    else:
        return speed, frs, pos

# ------------ plot ------------

# def main(model='kdsvp', k={}, tau={}, plot_3d=False):
#     speed, frs, kins = do_directions(model, k, tau)
#     non_directional, directional = \
#                      calculate.separate_directional_components(frs)
#     display.plot_components(non_directional, directional)
#     # masking okay to here
#     pds, ks, binsize, n_calcs = calculate.calc_moving_window_pds(frs)
#     display.plot_moving_window_pds(pds, ks, binsize,
#                                    n_calcs, non_directional.shape)
#     if plot_3d:
#         display.display_pds_3d_on_sphere(pds)
#     return pds, ks


#     def generate_progress_pattern():
#     '''
#     Generates a progress-only dependent signal (could be any function)
#     that is constant with constant fractional progress towards the target.
#     '''
#     pass

# def generate_firing_rate(pos, drn, speed, acc, time,
#                          PD, PG, k={}, tau={}, model='kd'):
#     ''' Generate simulated firing rate of a single cell during a single reach.

#     Parameters
#     ----------
#     pos : array_like, shape (n_bins, 3)
#         x,y,z position
#     drn : array_like, shape (n_bins - 1, 3)
#         x,y,z direction
#     speed : array_like, shape (n_bins,)
#         speed
#     acc : array_like, shape (n_bins - 1,)
#         acceleration
#     time : array_like, shape (n_bins)
#         time
#     PD : array_like, shape (3,) OR shape (n_bins - 1, 3) for changing PD
#         prefered direction
#     PG : array_like, shape (3,)
#         positional gradient
#     k : dictionary
#         scale factors for components of firing
#     tau : dictionary
#         lags for components of firing
#     model : string
#         see module docstring for details
        
#     Returns
#     -------
#     fr : masked_array_like, shape (n_bins - 1)
#         firing rates
#     PD : array, shape (3,) or (n_bins, 3)
#         preferred direction(s) used for model
#         useful for comparing with recovered PDs from simulated data
#     '''
#     assert np.rank(pos) == np.rank(drn) == 2
#     assert pos.shape[1] == drn.shape[1] == 3
#     assert 1 == np.rank(speed) == np.rank(acc) == np.rank(time)
#     n_pts = pos.shape[0]
#     assert speed.shape == time.shape == (n_pts,)
#     assert drn.shape[0] == n_pts - 1
#     assert acc.shape == (n_pts - 1, )
#     if np.rank(PD) == 1:
#         assert (len(PD) == 3)
#     else:
#         assert PD.shape[1] == 3
#     assert len(PG) == 3
#     assert type(k) == type(tau) == dict
#     assert type(model) == str

#     fr = np.ma.zeros(n_pts - 1)

#     if 'F' in model:
#         # Fisher-function distribution
#         # find angle between PD and d
#         if np.rank(PD) == 1:
#             theta = np.arccos(np.dot(drn, PD))
#         else:
#             raise NotImplementedError, 'Sorry Fisher tuning and changing' \
#                 ' PDs has yet to be implemented'
#         # calculate h(theta, kappa) (kappa = k['F'])
#         h = ss.vmf_pde_centered(theta, k['F'])
    
#     if 'k' in model:
#         # constant term
#         fr += k['k']

#     if 'p' in model:
#         # positional gradient
#         pos_component = k['p'] * np.sum(PG[None,...] * pos[:-1,...], axis=1)
#         #... pos.shape = (n_bins, 3)
#         if tau['p'] != 0:
#             fr += np.roll(pos_component, tau['p'])
#             #... select only valid region
#             low = tau['p'] if tau['p'] > 0 else None
#             high = tau['p'] if tau['p'] < 0 else None
#             mask = np.ones(n_pts - 1, dtype=bool)
#             mask[low:high] = False
#             fr.mask = mask
#         else:
#             fr += pos_component

#     if 'd' in model:
#         if 'F' in model: # Fisher tuning
#             fr += k['d'] * h
#             #does this scale okay? - from 0 at great distance to..? I think so.
#         else: # cosine tuning
#             # directional component
#             # have static PD
#             if np.rank(PD) == 1:
#                 fr += k['d'] * np.sum(PD[None,...] * drn, axis=1)
#             else:
#                 fr += k['d'] * np.sum(PD * drn, axis=1)
#             # PD[...,None].shape == (3,1), drn.shape == (3,199),
#             #   time.shape == 200

#     if 'v' in model:
#         # gain-field modulated speed (i.e. speed dot (direction from PD))
#         if 'F' in model: # Fisher tuning
#             fr += k['v'] * speed[:-1] * h
#         else: # cosine tuning
#             if np.rank(PD) == 1:
#                 v_comp = k['v'] * speed[:-1] * np.sum(PD[None,...] * drn, axis=1)
#                 fr += v_comp
#                 #print "Old way: %4f, %4f, %4f" % \
#                 #    (speed.max(), v_comp.max(),
#                 #     np.sum(PD[None,...] * drn, axis=1).max())
#             else:
#                 fr += k['v'] * speed[:-1] * np.sum(PD * drn, axis=1)

#     if 's' in model:
#         # additive speed component
#         fr += k['s'] * speed[:-1]

#     if 'n' in model:
#         # noise
# 	fr[fr < 0] = 1e-8
# 	mask = fr.mask.copy()
# 	fr[:] = poisson.rvs(fr * k['n'], size=fr.shape) / float(k['n'])
# 	fr.mask = mask
        
#     return fr

# def do_tasks(targets=None, model='kds',
#              k={}, tau={},
#              PD=None, PG=None,
#              full_output=True):
#     '''Creates firing rates for each direction requested.

#     Parameters
#     ----------
#     targets : array_like (n, 6)
#       array of start positions and targets,
#       n lots of [x_s, y_s, z_s, x_t, y_t, z_t]
#     model : string
#       code for model to use, see module docstring for description
#     k : dictionary
#       coefficients of firing rate components
#     tau : dictionary
#       lags of firing rate components

#     Returns
#     -------
#     time : ndarray, shape = 
#     speed : ndarray, shape = 
#     dirnames : list
#       string codes for each direction, in order found in frs
#     frs, masked array, shape = (n_dirs, n_pts - 1)
#       firing rates in each of the directions
#     all_pos : ndarray, shape = (n_dirs, n_pts, 3)
#       positions during movements in each of the directions
#     '''
#     assert type(targets) == np.ndarray
#     assert type(model) == str
#     assert type(k) == type(tau) == dict

#     k_defaults = {'k' : 1., 'p' : 1.,
#                   'd' : 1., 'v' : 1.,
#                   's' : 1., 'n' : 100,
#                   'F' : 1.}
#     k = myutils.dparse(k_defaults, k)
#     tau_defaults = {'p' : 0}
#     tau = myutils.dparse(tau_defaults, tau)
    
#     # global properties
#     time, disp, speed, acc = generate_ballistic_profiles(n_pts=100)
#     n_pts = time.size
#     n_dirs = targets.shape[0]

#     # preferred direction
#     if PD == None:
#         PD = generate_cell_property('PD')
#     if PG == None:
#         PG = generate_cell_property('PG')
#     if ('X' in model):
#         PDb = generate_cell_property('PDb')
#         PD = generate_changing_PD(PD, PDb, n_pts - 1)
#         pds = PD.repeat(n_dirs).reshape(n_pts - 1, 3, n_dirs).transpose([2,0,1])
#     else:
#         pds = PD.repeat(n_dirs * (n_pts - 1)).reshape(3, n_dirs, n_pts - 1).transpose([1,2,0])
        
#     # initialize storage variables
#     frs = np.ma.empty((n_dirs, n_pts - 1))
#     all_pos = np.empty((n_dirs, n_pts, 3)) + np.nan
#     starts = np.empty((n_dirs, n_pts, 3)) + np.nan
#     stops = np.empty((n_dirs, n_pts, 3)) + np.nan

#     # loop over targets
#     i_dir = 0
#     for target in targets:
#         start, stop = target[:3], target[3:]
#         starts[i_dir] = start[...,None].repeat(n_pts, axis=1).T
#         stops[i_dir] = stop[...,None].repeat(n_pts, axis=1).T
#         pos, direc = calc_pos_and_dir_along_movement(disp,
#                                                      start=start, stop=stop)
#         all_pos[i_dir] = pos
# 	if ('N' in model) and ('k' in model):
#             k['k'] *= np.random.normal(loc=1., scale=k['N'])
#         speed /= speed.max()
#         this_frs = generate_firing_rate(pos, direc, speed, acc, time,
#                                         PD, PG, k=k, tau=tau, model=model)
#         frs[i_dir] = this_frs
#         i_dir += 1
#     if full_output:
#         return time, speed, frs, all_pos, pds, starts, stops
#     else:
#         return time, speed, frs, all_pos

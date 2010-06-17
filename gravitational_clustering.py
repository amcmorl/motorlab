import numpy as np
from typechecking import accepts, NoneType
from vectors import norm

tau = 0.01 # 10 ms decay constant of charge
sigma = 350 # coalescence coefficient
resolution = 1e-3 # seconds per bin

@accepts(int)
def init_particles(N):
    '''
    Initialize particles.

    Each unit `i` starts at x_i = (x_ij : np.sqrt(2)/2.,)
                                  (x_i(^j) : 0.         )
    
    Parameters
    ----------
    N : int
      number of particles/neurons
    Returns
    -------
    xi : ndarray
      array of particle positions
    '''
    xi = np.zeros((N, N), dtype=float)
    loc = np.sqrt(2)/2.
    xi[np.eye(N, dtype=bool)] = loc
    return xi

@accepts(np.ndarray)
def distances(xi):
    '''
    Calculate distances between all particles.

    Parameters
    ----------
    xi : ndarray
      array of particle positions, shape=(particles, dims)=(N,N)

    Returns
    -------
    dist : ndarray
      distances between particles, should probably ignore diagonal
    '''
    # there are N**2 distances
    # construct a cube of comparisons
    # dist_vecs = (None, ptls, dims) - (ptls, None, dims) = (ptls, ptls, dims)
    dist_vecs = xi[None] - xi[:,None,:]
    dist = norm(dist_vecs, axis=2)
    return dist

def distances_over_time(xit):
    '''
    Calculate distances between all particles.

    Parameters
    ----------
    xi : ndarray
      array of particle positions, shape=(particles, dims)=(N,N)

    Returns
    -------
    distt : ndarray
      distances between particles, should probably ignore diagonal
    '''
    T, N, N = xit.shape
    distt = np.zeros(xit.shape)
    for t in xrange(T):
        distt[t] = distances(xit[t])
    return distt
    
decay = lambda q0, t, tau: q0 * np.exp(-t/tau)

def spikes_to_charge(spikes):
    """
    From list of spike times `spikes` construct the charge history
    
    Parameters
    ----------
    spikes : sequence
      sequence of spike times, length K

    Returns
    -------
    q : ndarray
      charges on particle instantaneously _after_ spikes,
      i.e. at times given by spikes
    
    Notes
    -----
    This is useful for offline data, where the entire spike train is known
    ahead of time. For online, use a different function.

    Most parsimonious, simple algorithm seems to be to calculate charge after
    each spike, so will construct len(K) array of charges at `spike` times.
    """
    spikes = np.asarray(spikes)
    assert(np.rank(spikes) == 1)
    
    q = np.zeros_like(spikes)
    q[1:] = np.diff(spikes)
    mean_interval = np.mean(q[1:])
    q[0] = mean_interval
    for b in xrange(1, spikes.shape[0]):
        q[b] = decay(q[b-1], spikes[b] - spikes[b-1], tau) + mean_interval

    # the non-for-loop way _may_ be developed from the sequence
    # q3 = DE1E2E3 + DE2E3 + DE3 + D
    # where D is delta, Ek is exp(-(t_k - t_{k-1})/tau)
    return q

def interpolate_q(q, times, nbins, trange=None):
    '''
    Parameters
    ----------
    q : ndarray
      charges, shape=(K,)
    times : sequence
      spike times (i.e. times for q), shape=(K,)
    nbins : int
      number of bins to interpolate into
    range : tuple | None
      if a tuple values are (min, max) of time range to interpolate between
    '''
    assert(type(q) == np.ndarray)
    times = np.asarray(times)
    assert(q.shape == times.shape)
    assert(type(nbins) == int)
    assert((type(trange) == tuple) or (type(trange) == NoneType))
    if trange == None:
        trange = (np.min(times), np.max(times))
    assert(len(trange) == 2)

    K = q.shape[0]
    # for each bin, go back to last q time and calculate charge at bin time
    bin_time = np.linspace(trange[0], trange[1], num=nbins, endpoint=True)
    
    gts = (bin_time[:,None] >= times[None,:])
    gt_idx = gts.astype(int) * (np.arange(K)[None,:] + 1)
    gest_idx = np.argmax(gt_idx, axis=1)
    haspreceder = np.max(gt_idx, axis=1) > 0

    lastq = np.zeros_like(bin_time)
    lastq[haspreceder] = q[gest_idx[haspreceder]]
    lastt = np.zeros_like(bin_time)
    lastt[haspreceder] = times[gest_idx[haspreceder]]
    qi = np.zeros(nbins, dtype=float)
    for idx, t in enumerate(bin_time):
        # get charge at last augmentation time
        qi[idx] = decay(lastq[idx], t - lastt[idx], tau)
    return qi

def construct_qit(allspikes):
    '''
    Construct ndarray with charge profiles over time for all units.
    
    Parameters
    ----------
    allspikes : list of list of float
      list of spike times for each unit, "shape" is N units x K spikes
    resolution : float
      binsize in seconds

    Returns
    -------
    qit : ndarray
      charges over time for each particle. shape=(N, B) where B is
      determined by time range of spikes, to give desired resolution
    T : ndarray
      times of time bins, shape B
    '''
    assert type(allspikes) == list
    mn = min([min(x) for x in allspikes])
    mx = max([max(x) for x in allspikes])
    nbins = int(np.round((mx - mn) / resolution, 0) + 1)
    qlist = [spikes_to_charge(sp) for sp in allspikes]
    qilist = [interpolate_q(q, sp, nbins, trange=(mn, mx)) \
                  for q, sp in zip(qlist, allspikes)]
    qit = np.asarray(qilist).T
    T = np.linspace(mn, mx, num=nbins, endpoint=True)
    return qit, T

def increment_position(xi, qi, h):
    """
    Parameters
    ----------
    xit : array_like
      particle positions at time t, shape (ptls, dims) i.e. (N,N)
    qi : array_like
      charge temporal profiles of particles, shape (ptls)
    h : float
      time step between bins, in seconds
    
    Returns
    -------
    Pnew : ndarray
      new positions of particles at time t + `h`


    Notes
    -----
    relies on module variables tau and sigma
    """
    xi = np.asarray(xi)

    # propulsive force = vector between / distance
    distij = xi[None] - xi[:,None] # ptls, ptls, dims
    sij = norm(distij, axis=2)
    rij = distij / sij[...,None] # nans for i=j :)

    # field =  sum of propulsive forces
    qjrij = (qi[None,:,None] - tau) * rij
    fi = np.nansum(qjrij, axis=1)
    xinh = xi + h * sigma * (qi - tau) * fi
    return xinh

@accepts(list)
def run(allspikes):
    """    
    Parameters
    ----------
    allspikes : list of lists of floats
      spike times for N units

    Returns
    -------
    xit : ndarray
      positions of particles over time, shape=(N,T)
    """
    errsets = np.seterr(invalid='ignore')
    N = len(allspikes)
    xi = init_particles(N)
    qit, T = construct_qit(allspikes)
    h = np.mean(np.diff(T))
    xit = np.zeros((T.shape[0],N,N))
    for t in xrange(T.shape[0]):
        xi = increment_position(xi, qit[t], h)
        xit[t] = xi
    np.seterr(**errsets)
    return xit

# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt

def plot_distances_over_time(xit):
    '''
    Parameters
    ----------
    Pt : array_like
      particle positions over time, shape=(ptls, dims, t)
    '''
    T, N, N = xit.shape
    distt = distances_over_time(xit)
    x, y = np.indices((N, N))
    validx = x > y
    plt.plot(distt[:,validx])

# ----------------------------------------------------------------------------

N = 3   # number of neurons
dur = 8.5 # duration in seconds
Hz = 10 # average firing rate

def test_init_particles():
    xi = init_particles(N)
    assert(xi.shape == (N, N))
    assert(xi[np.eye(N, dtype=bool)] == np.sqrt(2)/2.)

def test_distances():
    xi = init_particles(N)
    assert(np.allclose(distances(xi)[~np.eye(N, dtype=bool)], 1.))

def test_distances_over_time():
    xi0 = init_particles(N)
    xi1 = init_particles(N)
    xit = np.concatenate((xi0[None], xi1[None]))
    distt = distances_over_time(xit)
    mask = ~np.eye(N, dtype=bool)
    assert(np.all(distt[0, mask] == 1.))
    assert(np.all(distt[1, mask] == 2.))

def test_spikes_to_charge():
    spikes = [0.,1.,2.]
    q = spikes_to_charge(spikes)
    assert(q == np.ndarray([1., 1., 1.]))
    spikes = [0.,0.1,2.]
    q = spikes_to_charge(spikes)
    assert(np.all(q == np.array([1., 1 + np.exp(-1), 1.])))

def test_interpolate_q():
    spikes = [0., 1.]
    q = spikes_to_charge(spikes)
    nbins = 100
    qi = interpolate_q(q, spikes, nbins)
    t = np.linspace(0, 1, num=nbins, endpoint=True)
    actual = np.exp(-t[:-1]/tau)
    assert(np.allclose(actual, qi[:-1])

def test_construct_qit():
    nspikes = np.random.randint(1, dur * Hz, size=N)
    random_spikes = [list(np.sort(np.random.rand(x) * dur)) for x in nspikes]
    return random_spikes

# ----------------------------------------------------------------------------

@accepts(list)
def calc_mean_interval(spikes):
    '''
    From list of spike times, calculate the mean interval for each neuron.

    Parameters
    ----------
    spikes : list of list of float
      spikes times for each neuron, "shape" = N x K (different K for each n)
    
    Returns
    -------
    misi : ndarray
      mean interval for each neuron, shape=(N,)
    '''
    N = len(spikes)
    misi = np.zeros(N)
    for i, train in enumerate(spikes):
        misi[i] = np.mean(np.diff(train))
    return misi

def test_calc_mean_interval():
    spikes = [[0.,1.,2.],[0.,2.,4.]]
    misi = calc_mean_interval(spikes)
    assert(np.all(misi == np.array([1, 2])))


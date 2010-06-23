import numpy as np
from typechecking import accepts

def correlogram(x, y, trange=(-1e5, 1e5), nbins=100):
    '''Compute the correlogram between two spike trains.

    Parameters
    ----------
    x, y : array_like
      1d spike trains

    Returns
    -------
    xcor : cross-correlogram of spike trains `x` and `y`

    See Also
    --------
    NeuroTools.signals.analysis.ccf, which uses FFTs to calculate a fast
    cross-correlogram.
    
    Notes
    -----
    Adapted closely from implementation in neuropy, by Martin Spacek.
    '''
    x = np.asarray(x)
    y = np.asarray(y)

    dts = []
    for spike in x:
        trangei = np.searchsorted(y, spike + trange)
        dt = y[trangei[0]:trangei[1]] - spike
        dts.extend(dt)
    return np.histogram(dts, bins=nbins)

@accepts(list, float)
def threshold_by_rate(trains, threshold):
    '''
    Generate boolean mask of elements of `trains` with mean rate greater than
    `threshold` Hz.

    Parameters
    ----------
    trains : list of list of float
      list of spike trains (length N), each being a list of spike times
    threshold : float
      firing rate threshold to use

    Returns
    -------
    mask : ndarray
      elements of `trains` with mean rates gt `threshold`, shape=(N,)
    '''
    mask = np.zeros(len(trains), dtype=bool)
    for i, train in enumerate(trains):
        mask[i] = 1/np.mean(np.diff(train)) > threshold
    return mask

# ----------------------------------------------------------------------------
def test_threshold_by_rate():
    spikes = [[0, 0.5, 1.],        # rate 2 Hz
              [0., 0.25, 0.5],     # rate 4 Hz
              [0., 0.3, 0.6, 0.9]] # rate 3.33 Hz
    assert(np.all(threshold_by_rate(spikes, 1) == [True, True, True]))
    assert(np.all(threshold_by_rate(spikes, 2.2) == [False, True, True]))
    assert(np.all(threshold_by_rate(spikes, 3.5) == [False, True, False]))
    assert(np.all(threshold_by_rate(spikes, 5) == [False, False, False]))

def _generate_random_spikes():
    nspikes = np.random.randint(1, dur * Hz, size=N)
    random_spikes = [list(np.sort(np.random.rand(x) * dur)) for x in nspikes]
    return random_spikes

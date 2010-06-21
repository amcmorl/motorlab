import numpy as np

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

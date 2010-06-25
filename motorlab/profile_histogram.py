'''
Various forms of histogram for already sorted data,
with capacity to profile their speed.
'''
import numpy as np
import time
from scipy import weave
from fast_histogram import fast_hist

def make_spiketimes(rate=10., trange=np.array([0, 1])):
    trange = np.asarray(trange)
    duration = trange[1] - trange[0]
    
    interspike = np.random.exponential(1/rate, size=(rate * duration * 3))
    spike = np.cumsum(interspike) + trange[0]
    return spike[spike < trange[1]]

def inline_as_py(values, bins=10, range=None):
        # define bins, size N
    if (range is not None):
        mn, mx = range
        if (mn > mx):
            raise AttributeError(
                'max must be larger than min in range parameter.')

    if not np.iterable(bins):
        if range is None:
            range = (values.min(), values.max())
        mn, mx = [mi+0.0 for mi in range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = np.linspace(mn, mx, bins+1, endpoint=True)
    else:
        bins = np.asarray(bins)
        if (np.diff(bins) < 0).any():
            raise AttributeError(
                'bins must increase monotonically.')
    
    # define n, empty array of size N+1
    count = np.zeros(bins.size - 1, int)
    nvalues = values.size
    nbins = bins.size

    if values.size == 0:
        raise AttributeError(
            'a must contain some data')
    
    if values[-1] < bins[0]:
        raise AttributeError(
            'last element of a must be smaller than first element of bins')

    if (values[0] > bins[0]):
        rb = 0;
    else:
        lb = 0;
        rb = nvalues + 1;
        while(lb < rb - 1):
            if (values[(lb + rb) / 2.] < bins[0]):
                lb = (lb + rb) / 2.
            else:
                rb = (lb + rb) / 2.

    # Sweep through the values, counting, until they get too big
    lb = 0;
    valid = (rb < nvalues)
    if valid:
        valid = valid & (values[rb] < bins[nbins - 1])
    while valid:
        # Advance the edge caret until the current value is in the current bin
        while (bins[lb+1] < values[rb]):
            lb += 1
        # Increment the current bin
        count[lb] += 1
        # Increment the value caret
        rb += 1
        valid = (rb < nvalues)
        if valid:
            valid = valid & (values[rb] < bins[nbins - 1])

    return count, bins

def is_correct(hist):
    if np.all(hist == np.array([1,2,1,0,1,2,0,0,1,1])):
        return True
    else:
        return False

def main():
    spikes = make_spiketimes(trange=(0,1e4))

    #spikes = np.array([0.01, 0.12, 0.14, 0.21, 0.42, 0.56, 0.57, 0.81, 0.99])
    bins = np.linspace(0, 1, 11, endpoint=True)
    methods = [np.histogram, inline, inline_as_py]
    times = []
    for fn in methods:
        t = time.clock()
        hist, bins = fn(spikes, bins)
        dt = time.clock() - t
        print "Procedure: %s" % (fn.__name__)
        print "-----------" + "-" * len(fn.__name__)
        print "Counts: ", hist, 
        #print "is %scorrect" % ("" if is_correct(hist) else "NOT ")
        print "Time: %0.5f" % (dt)
        print ""
        times.append(dt)
    return times


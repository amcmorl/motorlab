import numpy as np
from matplotlib.pyplot import draw

def plot_raster_list(spike_series, ax):
    '''Plots a series of rasters in the given axes.
    Accepts a list of spike trains.

    Parameters
    ----------
    spike_series : list of array_like
      1st dim is number of tasks, each item in list is an array of spike times
    ax : matplotlib axis
    '''
    ntrains = len(spike_series)
    lines = []
    for i, spikes in enumerate(spike_series):
        if len(spikes) > 0:
            lines.append(plot_raster_row(spikes, ax, i))
    ax.set_ylim(-0.5, ntrains - 0.5)
    ax.set_yticks(np.arange(ntrains))
    ax.set_yticklabels([str(x) for x in xrange(ntrains)])
    draw()
    return lines

def plot_raster_row(times, ax, y):
    '''
    Plots a single row of raster ticks.

    Parameters
    ----------
    times : 1d array-like of float
      spike times to mark
    ax : matplotlib axis object
      axis to plot on
    ymin : float
      min value to draw ticks from
    ymax : float
      max value to draw ticks to
    '''
    if len(times) > 0:
        times = np.asarray(times)
        ys = np.ones_like(times) * y
        line = ax.plot(times, ys, '.', markersize=2., markeredgewidth=0.)
        return line

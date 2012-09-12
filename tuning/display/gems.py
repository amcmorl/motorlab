import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec, \
    GridSpecFromSubplotSpec

import split_lambert_projection

from proc_tools import rolling_window
from vectors import norm
from spherical import cart2pol
from motorlab.tasks import make_26_targets

def plot_gem(data, order='co', fig=None, ax=None,
             clim=None, **kwargs):
    '''
    Parameters
    ----------
    data : ndarray
      shape (26,)
    order : string, optional
      one of 'co' or 'oc', for center-out or out-center data order
      determines mapping of data to polygons
      defaults to center-out
    fig : matplotlib Figure instance
      an existing figure to use, optional
    ax : SplitLambertAxes instance
      an existing axis to use, optional
    clim : tuple or list of float, optional
      minimum and maximum values for color map
      if not supplied, min and max of data are used
    '''
    if not data.shape == (26,):
        raise ValueError('data has wrong shape; should be (26,),'
                         ' actually is %s' % (str(data.shape)))
    if order == 'co':
        mapping = get_targ_co_dir_mapping()
    elif order == 'oc':
        mapping = get_targ_oc_dir_mapping()
    else:
        raise ValueError('`order` not recognized')
    if (fig != None) & ~isinstance(fig, Figure):
        raise ValueError('`fig` must be an instance of Figure')
    if (ax != None) & \
            ~isinstance(ax, split_lambert_projection.SplitLambertAxes):
        raise ValueError('`ax` must be an instance of SplitLambertAxes')
    if (clim != None):
        if (type(clim) != tuple) and (type(clim) != list):
            raise ValueError('clim must be tuple or list')
        if len(clim) != 2:
            raise ValueError('clim must have 2 elements')
    else:
        clim = (np.min(data), np.max(data))
        
    if (fig == None):
        if (ax != None):
            fig = ax.figure
        else:
            fig = plt.figure()
    if (ax == None):
        ax = fig.add_subplot(111, projection='split_lambert')
    
    pts, spts, circs = make_26_targets()
    pts = pts.astype(float)
    pts /= norm(pts, axis=1)[:,None]
    tps = cart2pol(pts)

    q = 9
    cnrs = np.zeros((32, 5, 2)) # for non-polar polys
    ends = np.zeros((2, q, 2))  # for poles
    
    # for now use boundaries suggested by square ends
    squares = np.unique(tps[circs[0]][:,0])
    mids = np.mean(rolling_window(squares, 2), axis=1)
    eps = 1e-8
    tvs = np.hstack([mids[:2], np.pi / 2. - eps, np.pi / 2., mids[-2:]])
    pvs = np.linspace(0, 2 * np.pi, q, endpoint=True) % (2 * np.pi) \
        - np.pi / 8.

    k = 0
    rings = [0,1,3,4]
    for i in rings: # 3 rings of 4-cnr polys
        for j in xrange(8): # 8 polys per ring
            xs = np.array((tvs[i], tvs[i], tvs[i+1], tvs[i+1], tvs[i]))
            ys = np.array((pvs[j], pvs[j+1], pvs[j+1], pvs[j], pvs[j]))
            cnrs[k] = np.vstack((xs, ys)).T
            k += 1

    ends[0,:,0] = np.ones(q) * tvs[0]
    ends[1,:,0] = np.ones(q) * tvs[-1]
    ends[0,:,1] = pvs
    ends[1,:,1] = pvs

    kwargs.setdefault('edgecolors', 'none')
    kwargs.setdefault('linewidths', (0.25,))
    coll_cnrs = mcoll.PolyCollection(cnrs, **kwargs)
    coll_ends = mcoll.PolyCollection(ends, **kwargs)
    
    mapped_data = data[mapping]

    coll_cnrs.set_array(mapped_data[:-2])
    coll_ends.set_array(mapped_data[-2:])

    if clim == None:
        cmin, cmax = mapped_data.min(), mapped_data.max()
    else:
        cmin, cmax = clim
    coll_cnrs.set_clim(cmin, cmax)
    coll_ends.set_clim(cmin, cmax)
    
    ax.add_collection(coll_cnrs)
    ax.add_collection(coll_ends)
    return ax

def get_targ_co_dir_mapping():
    '''
    Get mapping from center-out directions
    (as per dc.tasks) to split lambert target plot polygons
    '''
    return [2,5,10,17,22,19,14,7,0,1,9,21, \
                25,24,16,4,0,1,9,21,25,24,16, \
                4,3,6,11,18,23,20,15,8,12,13]

def get_targ_oc_dir_mapping():
    '''
    Get mapping from center-out directions
    (as per dc.tasks) to split lambert target plot polygons
    '''
    return [23,20,15,8,3,6,11,18,25,24,16,4, \
                0,1,9,21,25,24,16,4,0,1,9, \
                21,22,19,14,7,2,5,10,17,13,12]

def plot_both_directions(data, co, fig=None):
    '''
    Parameters
    ----------
    data : ndarray
      shape (52=ntask, nbin)
    co : ndarray
      shape (52,), dtype bool, array of which tasks are center-out
    fig : matplotlib Figure instance
      an existing figure to use, optional
    '''
    if not ((np.rank(data) == 2) & (data.shape[0] == 52)):
        raise ValueError('data has wrong shape; should be (52, nbin),'
                         'actually is %s' % (str(data.shape)))
    if (fig != None) & (not isinstance(fig, Figure)):
        raise ValueError('`fig` must be an instance of Figure')
    if (fig == None):
        fig = plt.figure()
    gs = GridSpec(1,2)
    gs.update(left=0.02, right=0.98, top=0.98, bottom=0.05, hspace=0.02)
    plot_series(data[co], order='co', fig=fig, subplot_spec=gs[0])
    plot_series(data[~co], order='oc', fig=fig, subplot_spec=gs[1])

def plot_series(data, order='co', fig=None, subplot_spec=None):
    '''
    Parameters
    ----------
    data : ndarray
      shape (26=ntask, nbin)
    order : string, optional
      one of 'co' or 'oc', for center-out or out-center data order
      determines mapping of data to polygons
      defaults to center-out
    fig : matplotlib Figure instance
      an existing figure to use, optional
    subplotspec : matplotlib SubplotSpec instance, optional
      an existing subplotspec to use, i.e. subset of a gridspec
    '''
    if not ((np.rank(data) == 2) & (data.shape[0] == 26)):
        raise ValueError('data has wrong shape; should be (26, nbin),'
                         'actually is %s' % (str(data.shape)))
    if order == 'co':
        mapping = get_targ_co_dir_mapping()
    elif order == 'oc':
        mapping = get_targ_oc_dir_mapping()
    else:
        raise ValueError('`order` not recognized')
    
    if (fig != None) & (not isinstance(fig, Figure)):
        raise ValueError('`fig` must be an instance of Figure')
    if (fig == None):
        fig = plt.figure()
    if (subplot_spec != None):
        if not isinstance(subplot_spec, SubplotSpec):
            raise ValueError('subplot_spec must be instance of '
                             'SubplotSpec')
    ntask, nbin = data.shape
    clim = (np.nanmin(data), np.nanmax(data))
    if subplot_spec == None:
        gs = GridSpec(nbin + 1,1, height_ratios=[1,] * nbin + [0.5,])
        gs.update(left=0.02, right=0.98, top=0.98, bottom=0.05)
    else:
        gs = GridSpecFromSubplotSpec(\
            nbin + 1,1, subplot_spec=subplot_spec, \
                height_ratios=[1,] * nbin + [0.5,])
    for i in xrange(nbin):
        ax = fig.add_subplot(gs[i], projection='split_lambert')
        plot_gem(data[:,i], order=order, ax=ax, clim=clim)
    cbax = fig.add_subplot(gs[-1], aspect=.25)
    cb = plt.colorbar(ax.collections[0], cbax, orientation='horizontal')
    clim = cb.get_clim()
    cb.set_ticks(clim)
    cb.set_ticklabels(['%0.1f' % x for x in clim])
    return fig



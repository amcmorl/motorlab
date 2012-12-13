import numpy as np
from amcmorl_py_tools.vecgeom import unitvec
from amcmorl_py_tools.vecgeom.rotations import rotate_by_angles
from amcmorl_py_tools.vecgeom.coords import pol2cart, cart2pol
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from warnings import warn
from motorlab.tuning import sort
import amcmorl_py_tools.plot_tools as ptl

def _map_data(data, task, pd, resolution=100):
    '''
    Parameters
    ----------
    data : ndarray
      shape (ntime, ntask)
    task : ndarray
      shape (ntask, 6)
    pd : ndarray
      shape (3,)
      
    Returns
    -------
    mapped_data : ndarray
      shape (ntime, resolution, resolution)

    Notes
    -----
    need to incorporate PD so that it lies at center of plot
    '''
    if data.shape[-1] != task.shape[0]:
        raise ValueError('last axis of data (len %d) must be same length as\n'
                         'first axis of task (len %d)' % \
                             (data.shape[-1], data.shape[0]))
    
    # generate theta, phi grid
    # or rather map x,y grid to nearest of 26 targets
    thetas = np.linspace(0., np.pi, resolution)
    phis = np.linspace(0., 2 * np.pi, resolution)
    theta_grid = np.tile(thetas[:,None], (1, resolution))
    phi_grid = np.tile(phis[None,:], (resolution, 1))

    # convert polar to cartesian co-ordinates
    tp_grid = np.concatenate((theta_grid[None], phi_grid[None]), axis=0)
        
    xyz_grid = pol2cart(tp_grid, axis=0)    

    # calculate direction of each task
    start = task[:,:3]
    stop = task[:,3:]
    direction_task = unitvec(stop - start, axis=-1)

    # rotate task directions until pd points towards 0,0,1
    rotated_toz = rotate_targets(direction_task, cart2pol(pd))

    # now rotate again to point pd towards 0,1,0
    rotated_toy = rotate_targets(rotated_toz, np.array([np.pi/2., 0]))
    
    # calculate angle between each grid square and each direction
    angles = np.arccos(np.tensordot(rotated_toy, xyz_grid, [1,0]))

    # get index of closest direction for each grid square
    # = get argmin along 0th axis
    nearest = np.argmin(angles, axis=0)

    mapped_data = data[..., nearest]
    return mapped_data

def rotate_targets(directions, theta_phi):
    return np.apply_along_axis(rotate_by_angles, 1, directions,
                               theta_phi[0], theta_phi[1], True)

def plot_many_azel(data, task, pds, ncol=4,
                   labels=None, figsize=None,
                   **kwargs):
    nazel = data.shape[0]
    #print "Figsize", figsize
    fig = plt.figure(figsize=figsize)
    if nazel < ncol:
        ncol = nazel
        nrow = 1
    else:
        nrow = nazel / ncol + int(nazel % ncol != 0)
        # add 1 [== int(True)] to nrow
        #   if ncol doesn't divide evenly into nazel
    gs = GridSpec(nrow, ncol)
    for i in xrange(data.shape[0]):
        plot_data = data[i]
        pd = pds[i]
        if labels != None:
            if not len(labels) == nazel:
                raise ValueError('labels must be list of length data.shape[0]')
            label = labels[i]
        else:
            label = ''
        plot_azel(plot_data.T, task, pd, 
                  subplot_spec=gs[i], fig=fig, label=label,
                  cbar=False, **kwargs)
    return fig, gs

def plot_azel(data, task, pd, resolution=100, cbar=True,
              subplot_spec=None, fig=None, label='', vscale=None,
              labelpad=0.05):
    '''
    Parameters
    ----------
    data : ndarray
      shape (ntime, ntask)
    task : ndarray
      shape (ntask, 6)
    pd : ndarray
      shape (3,)
    resolution : int
      number of pixels for azel images
    cbar : bool
      draw a colour bar?
    subplot_spec : matplotlib SubplotSpec or None
      a SubplotSpec in which to draw the plot
      a `fig` must also be supplied
    fig : matplotlib Figure or None
      only used if subplot_spec is not None
      existing figure in which to draw plot
    vscale : tuple (float, float)
      min and max color scale values
    labelpad : float
      spacing between top of plots and label
    '''
    assert(np.rank(data) == 2)
    assert(data.shape[1] == task.shape[0])
    assert(pd.shape == (3,))
    assert(type(resolution) == int)
    assert((subplot_spec == None) == (fig == None))
    assert((vscale == None) | ((type(vscale) == tuple)))
    if type(vscale) == tuple:
        assert((len(vscale) == 2))
        assert(np.isscalar(vscale[0]))
    assert((type(cbar) == bool) | (cbar == 'ghost'))
    
    ntime = data.shape[0]
    if vscale != None:
        vmin = vscale[0]
        vmax = vscale[1]
    else:
        vmin = np.min(data)
        vmax = np.max(data)
    co = sort.get_center_out(task)

    height_ratios = [1,] * ntime
    if cbar:
        extra = 1
        height_ratios += [0.3,]
    else:
        extra = 0
        
    if subplot_spec == None:
        # make our own figure
        w = 1.5
        h = 7
        fig = plt.figure(figsize=(w, h))
        gs = GridSpec(ntime + extra, 2, hspace=0.0, wspace=0.0,
                      left=0, right=1., bottom=0,
                      height_ratios=height_ratios)
        top = gs[0,0].get_position(fig).y1
        left = gs[0,0].get_position(fig).x0
        right = gs[0,1].get_position(fig).x1
        center = (left + right) / 2.
    else:
        # subplot_spec != None
        #m = ptl.subplot_spec2margins(subplot_spec, fig)
        #ax_rects = ptl.get_ax_rects(np.arange(ntime * 2), 2, ntime,
        #                            margin=m, direction='col')
        gs = GridSpecFromSubplotSpec(ntime + extra, 2, hspace=0, wspace=0,
                                     subplot_spec=subplot_spec)
        #gs.update(left=0.1, right=0.9, bottom=0.1, hspace=0.05, wspace=0.05)
        gs.set_height_ratios(height_ratios)
        box = subplot_spec.get_position(fig)
        top = box.y1
        center = (box.x0 + box.x1) / 2.
    
    for i, which in enumerate([co, ~co]):
        mapped_data = _map_data(data[:,which], task[which], pd, resolution)
        for j in xrange(ntime):
            ax = fig.add_subplot(gs[j,i])
            #ax = fig.add_axes(ax_rects[i * ntime + j])
            ax.imshow(mapped_data[j], vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
    
    if label != '':
        fig.text(center, top + labelpad, label, fontsize='large', ha='center')

    # explicit boolean check to avoid cbar == 'ghost' case
    if cbar == True:
        cbax = fig.add_subplot(gs[-1,:], aspect=0.15)
        cb = plt.colorbar(ax.images[0], cbax, orientation='horizontal')
        clim = cb.get_clim()
        cb.set_ticks(clim)
        cb.set_ticklabels(['%0.1f' % x for x in clim])

    return fig

def plot_array_azel(bnd, arr, vscale='each'):
    '''
    Plot azels of data from an array, with an arbitrary PD.
    
    Parameters
    ----------
    bnd : BinnedData
      used for tasks only
    score : ndarray
      data to plot, shape (ndata, ntask, nbin)
    vscale : string
      if 'each', use cmap from each data set separately
      if 'all', use min and max of total arr to colour
    '''
    pd = np.array([0.,1.,0.]) # default for no rotation (???)
    nscore = arr.shape[0]
    pds = np.tile(pd[None], (nscore, 1))
    labels = ['PC %d' % (x + 1) for x in xrange(nscore)]
    if vscale == 'each':
        vscale = None
    elif vscale == 'all':
        vscale = (np.nanmin(arr), np.nanmax(arr))
    fig, gs = plot_many_azel(arr, bnd.tasks, pds,
                             labels=labels, figsize=(6,8), vscale=vscale)
    return fig
    
    
    
    

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:57:44 2011

@author: amcmorl
"""
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from warnings import warn
import plot_tools as ptl

def minmax(arr, axis=0):
    return np.concatenate([np.min(arr, axis=axis)[:,None],
                           np.max(arr, axis=axis)[:,None]], axis=1)

def triter(dim):
    for x in xrange(dim[0]):
        for y in xrange(dim[1]):
            for z in xrange(dim[2]):
                yield x,y,z

def mean_at(vals, inds):
    # iterate over dimensions of inds i,j,k
    # getting mean of all values for rates[inds[i,j,k]]
    dim = inds.shape[1:]
    mean = np.zeros(dim) + np.nan
    for x,y,z in triter(dim):
        keep = inds[:,x,y,z]
        if np.any(keep):
            kept = vals[keep]
            mean[x,y,z] = np.mean(kept)
        else:
            mean[x,y,z] = np.nan
    return mean

def get_mapped(pos, rate, ranges=None, resolution=5):
    '''
    Parameters
    ----------
    pos : array_like
      shape (npt, 3), positions
    rate : array_like
      shape (npt,), firing rates
    ranges: array_like, optional
      shape (3, 2), max and min positions in each dimension
      if None, defaults to max and min of data
    resolution : int or sequence of int
      number of bins to use in each dimension
    '''
    # calculate max and min values
    if np.rank(pos) != 2:
        raise ValueError('pos must have 2 dimensions')
    if np.rank(rate) != 1:
        raise ValueError('rate must have 1 dimension')
    if ranges != None:
        ranges = np.asarray(ranges)
        if ranges.shape != (3,2):
            raise ValueError('ranges, if specified, must have shape (3,2)')
    if type(resolution) != int:
        if len(resolution) != 3:
            raise ValueError('resolution must be of type int, '
                             'or a list of length 3')
    else: # type(resolution) == int
        resolution = np.ones(3) * resolution
    
    if ranges == None:
        ranges = np.array(minmax(pos, axis=0))
        
    spread = np.diff(ranges, axis=1).astype(float)
                          
    # need two arrays of shape 3,x,y,z: one with min pts, one with max pts
    edge_shape = resolution + 1
    edges = np.mgrid[0:edge_shape[0],
                     0:edge_shape[1],
                     0:edge_shape[2]].astype(float)
    edges *= (spread[...,None,None] / resolution[:,None,None,None])
    edges += ranges[:,0][:,None,None,None]
    
    gt = pos[...,None,None,None] > edges[None,:,:-1,:-1,:-1]
    lt = pos[...,None,None,None] < edges[None,:,1:,1:,1:]
    jr = np.all(gt & lt, axis=1) # pts that fit all criteria, for each x,y,z
    mean = mean_at(rate, jr)
    
    # map exactly 0 values to nans
    # since >>>> likely no data was there
    return mean

def plot_position_rates(pos, rate, ranges=None, 
                        resolution=5,
                        subplot_spec=None, fig=None,
                        vmin=None, vmax=None):
    '''
    Plot grids showing y slices through space, each representing and x-z plane,
    colour-coded by firing rate.
    
    Parameters
    ----------
    pos : array_like
      shape (npt, 3), positions
    rate : array_like
      shape (npt,), firing rates
    ranges: array_like, optional
      shape (3, 2), max and min positions in each dimension
      if None, defaults to max and min of data
    resolution : int or sequence of int
      number of bins to use in each dimension
      
    Returns
    -------
    fig : matplotlib Figure containing the plot
    '''        
    mapped = get_mapped(pos, rate, ranges=ranges, resolution=resolution)
    nlevel = mapped.shape[1]

    if subplot_spec == None:
        m = ptl.Margins(left=0.1, bottom=0.1, right=0.1, top=0.1,
                        hgap=0., vgap=0.)
        ax_rects = ptl.get_ax_rects(np.arange(nlevel), nlevel, 1,
                                    margin=m, direction='row')
        #gs = GridSpec(1, nlevel)
        if fig == None:
            warn("ignoring fig object")
        fig = plt.figure()
    else: # subplot_spec != None
        m = ptl.subplot_spec2margins(subplot_spec, fig)
        ax_rects = ptl.get_ax_rects(np.arange(nlevel), nlevel, 1,
                                    margin=m, direction='row')
        #gs = GridSpecFromSubplotSpec(1, nlevel, subplot_spec=subplot_spec)
        #if fig == None:
        #    raise ValueError("when subplot_spec is set, "
        #                     "fig must be a Figure object")
        
    if vmin == None:
        vmin = np.nanmin(mapped)        
    if vmax == None:
        vmax = np.nanmax(mapped)
        
    for i in xrange(nlevel):
        #ax = fig.add_subplot(gs[l])
        #print ax_rects[i]
        ax = fig.add_axes(ax_rects[i])
        ax.imshow(mapped[:,i,:].T, aspect='auto',
                  vmin=np.nanmin(mapped), vmax=np.nanmax(mapped))
        ax.set_xticklabels([])
        xtl = ax.get_xticklines()
        for xt in xtl:
            xt.set_visible(False)
        ax.set_yticklabels([])
        ytl = ax.get_yticklines()
        for yt in ytl:
            yt.set_visible(False)
    return fig
        
def test_plot_position_rates_simulated_positions():
    pos = np.random.uniform(-0.6, 0.6, size=(1000,3))
    rate = np.exp(pos[:,2]) * 2.4
    fig = plot_position_rates(pos, rate)

if __name__ == "__main__":
    test_plot_position_rates_simulated_positions()
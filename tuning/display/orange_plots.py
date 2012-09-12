import numpy as np
import plot_tools
import matplotlib.pyplot as plt

tdelta = 0.005

def plot_orange(data, fig=None,
                margin=plot_tools.Margins(vgap=0.01, hgap=0.01),
                vmin=None, vmax=None, label=None, fontsize=14):
    '''Each orange plot has 8 panels: 4 segments center-out, 4 out-center
    '''
    ntask, nbin = data.shape
    nplot = 8
    nrow = 4
    ncol = 2
    #aspect = 38 / 42. # vertical / horizontal
    if ntask != 52:
        raise ValueError('dim 0 of `data` must be 52')
    
    if fig == None:
        fig = plt.figure(figsize=(6.63,6))
        
    idxs = [[26, 28, 29, 27, 25, 23, 22, 24], # co panel0 to +y...
            [26, 33, 37, 32, 25, 18, 14, 19], # panel1 to +x+y...
            [26, 36, 38, 35, 25, 15, 13, 16], # panel2 to +x...
            [26, 31, 34, 30, 25, 20, 17, 21], # panel3 to +x-y...
            [12, 10,  9, 11, 39, 41, 42, 40], # oc... 
            [12,  5,  1,  6, 39, 46, 50, 45], # same directions as above
            [12,  2,  0,  3, 39, 49, 51, 48],
            [12,  7,  4,  8, 39, 44, 47, 43]]
    
    rects = plot_tools.get_ax_rects(np.arange(nplot), ncol, nrow,
                                    margin=margin, direction='col')
    axes = []
    for rect, idx in zip(rects, idxs):
        ax = fig.add_axes(rect)
        if vmin == None:
            vmin = np.min(data)
        if vmax == None:
            vmax = np.max(data)
        ax.imshow(data[idx,:], aspect='auto',
                  vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
    if label != None:
        rect = margin.get_rect()
        center = rect[0] + rect[2] / 2.
        top = rect[1] + rect[3]
        fig.text(center, top + tdelta, label,
                 ha='center', fontsize=fontsize)
    return axes

def plot_orchard(data, fig=None,
                 margin=plot_tools.Margins(left=0.05, bottom=0.05,
                                           top=0.05, right=0.05,
                                           vgap=0.05, hgap=0.02),
                 submargin=plot_tools.Margins(vgap=0., hgap=0.),
                 label=None, fontsize=24, ncol=5,
                 vmin=None, vmax=None, own_color=False):
    '''
    Parameters
    ----------
    data : ndarray
      rate | score data, shape (nthing, ntask, nbin)
      where ntask should be 52
    '''
    if fig == None:
        fig = plt.figure(figsize=(15,10))
    
    ndset = data.shape[0]
    nrow = plot_tools.make_enough_rows(ndset, ncol)
    rects = plot_tools.get_ax_rects(np.arange(ndset), ncol, nrow,
                                    margin=margin, direction='row')
    oranges = []
    
    if not own_color:
        if vmin == None:
            vmin = np.min(data)
        if vmax == None:
            vmax = np.max(data)
    
    for idset, (dset, rect) in enumerate(zip(data, rects)):
        center = rect[0] + rect[2] / 2.
        top = rect[1] + rect[3]
        if label != None:
            fig.text(center, top + tdelta, label[idset],
                     ha='center', fontsize=fontsize)
        submargin.set_from_rect(rect)
        orange = plot_orange(dset, fig=fig, margin=submargin,
                             vmin=vmin, vmax=vmax)
        oranges.append(orange)
    return fig

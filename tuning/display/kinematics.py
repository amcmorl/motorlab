import numpy as np
from coord_primitives import sphere
import matplotlib.pyplot as plt
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib import cm    
from mv_tools import draw_sphere

def plot_profiles(time, disp, speed, acc, figure=1):
    """
    """
    fig = plt.figure(figure)

    ax_disp = fig.add_subplot(311)
    ax_disp.plot(time, disp, 'k-', label='displacement')
    ax_disp.set_xlabel('time')
    ax_disp.set_ylabel('displacement')

    ax_speed = fig.add_subplot(312)
    ax_speed.plot(time, speed, 'k-', label='speed')
    ax_speed.set_xlabel('time')
    ax_speed.set_ylabel('speed')

    ax_acc = fig.add_subplot(313)
    ax_acc.plot(time[:-1], acc, 'k-', label='acceleration')
    ax_acc.set_xlabel('time')
    ax_acc.set_ylabel('acceleration')

def plot_speed_alignment(cell, threshold=0.15):
    plt.figure(1)
    aut, adt, vts, (sp, ts) = cell.calc_movement_start_finish_times( \
        threshold=threshold, earlymax_limit=0.2, full_output=True)
    for i in xrange(len(sp)):
        offset = aut[i]
        au_time = aut[i] - offset
        ad_time = adt[i] - offset
        time = ts[i][:-1] - offset
        plt.plot(time, sp[i])
        plt.plot((time[np.argmax(sp[i])],), (sp[i].max(), ), 'bo')
        plt.axhline(threshold, color='k')
        plt.axvline(au_time, color='g')
        plt.axvline(ad_time, color='r')

    plt.figure(2)
    start_t, stop_t, valid_trials = cell.calc_clip_times()
    for i in xrange(len(sp)):
        #offset = start_t[i]
        time = ts[i][:-1]
        near_start_bin = np.argmin(np.abs(time - start_t[i]))
        near_stop_bin = np.argmin(np.abs(time - stop_t[i]))
        scaled_t = np.linspace(0, 1., near_stop_bin - near_start_bin)
        plt.plot(scaled_t, sp[i][near_start_bin:near_stop_bin])
    plt.show()

def plot_targets(tasks, color=(1., 0., 0.), scale_factor=0.005):
    '''Plot start and end points of tasks.

    Starting points are designated by spheres, and end points by cubes.
    '''
    # draw target positions
    p3d = mlab.pipeline
    fig = mlab.gcf()
    fig.scene.disable_render = True
    
    starts, stops = tasks[:,:3], tasks[:,3:]

    # plot start points
    x,y,z = starts.T
    start_src = p3d.scalar_scatter(x, y, z)
    gl1 = p3d.glyph(start_src)
    gl1.glyph.glyph.scale_factor = scale_factor         # smaller
    gl1.actor.actor.property.color = color # red
    gl1.actor.actor.property.opacity = 0.5 # red

    # plot stop points
    #x,y,z = stops.T
    #stop_src = p3d.scalar_scatter(x, y, z)
    #gl2 = p3d.glyph(stop_src)
    #gl2.glyph.glyph.scale_factor = scale_factor         # smaller
    #gl2.actor.actor.property.color = color # red
    #cube = gl2.glyph.glyph_source.glyph_dict['cube_source']
    #gl2.glyph.glyph_source.glyph_source = cube
    #gl2.actor.actor.property.opacity = 0.5

    fig.scene.disable_render = False
    return gl1

def construct_targets_mesh():
    '''
    Returns
    -------
    x,y,z : ndarrays

    '''
    sphere_pts = sphere(npts=(9,5))
    sp = np.asarray(sphere_pts)
    sp2 = sp.reshape(3, 5*9)
    wrong = np.abs(np.abs(sp2[0]) - 0.5) < 1e-8
    signs = np.sign(sp2[:,wrong])
    sp2[:,wrong] = signs * 1/np.sqrt(3.)
    x,y,z = sp
    return x,y,z,sp2

def plot_trajectories(pos, scalars=None, radius=0.0005, lut='blue-red',
                      cmin=None, cmax=None):
    '''Plot 3d traces of movements during trials.

    Parameters
    ----------
    pos : array, shape (n_dirs, n_reps, n_bins, n_dims)
    '''
    assert type(pos) == np.ndarray
    assert pos.shape[-1] == 3

    p3d = mlab.pipeline
    f = mlab.gcf()
    f.scene.disable_render = True

    if np.rank(pos) == 3:
        n_dirs, n_bins, n_dims = pos.shape
        pos = pos[:, None,...]
    n_dirs, n_reps, n_bins, n_dims = pos.shape
    if scalars == None:
        scalars = np.arange(n_dirs).repeat(n_reps * n_bins).reshape( \
            n_dirs, n_reps, n_bins)
    if cmin == None:
        cmin = scalars.min()
    if cmax == None:
        cmax = scalars.max()
    for i_dir in xrange(n_dirs):
        for i_rep in xrange(n_reps):
            x,y,z = pos[i_dir, i_rep].T
            pt_src = p3d.line_source(x,y,z,scalars[i_dir, i_rep])
            tube = p3d.tube(pt_src)
            tube.filter.radius = radius
            tube.filter.number_of_sides = 12
            surf = p3d.surface(tube)
            mm = surf.module_manager
            mm.scalar_lut_manager.lut_mode = lut
            if scalars != None:
                mm.scalar_lut_manager.default_data_range = \
                    np.array((cmin, cmax))
            else:
                surf.actor.property.color = (1.0 - i_dir/float(n_dirs - 1),
                                             0., i_dir/float(n_dirs - 1))
    f.scene.disable_render = False
    return f

def plot_trajectories_other(pos):
    '''
    '''
    import enthought.mayavi.mlab as mlab
    n_tasks, n_bins, n_dims = pos.shape
    f = mlab.gcf()
    f.scene.disable_render = True
    for i_task in xrange(n_tasks):
        x,y,z = pos[i_task,:,:].T
        surf = mlab.plot3d(x,y,z)
        #tube = surf.parent.parent
        surf.actor.property.color = (0., 0.3, 1.0)
        #tube.filter.radius = 0.75
    f.scene.disable_render = False
    return surf

    
def plot_trajectory_points(pos, color=(0., 0., 1.)):
    '''
    '''
    import enthought.mayavi.mlab as mlab
    n_tasks, n_bins, n_dims = pos.shape
    f = mlab.gcf()
    f.scene.disable_render = True
    for i_task in xrange(n_tasks):
        x,y,z = pos[i_task,:,:].T
        glyph = mlab.points3d(x,y,z)
        glyph.glyph.glyph.scale_factor = 0.01
        glyph.actor.actor.property.color = color
    f.scene.disable_render = False
    return glyph
    
def plot_xyz(xyz, xs=None, figsize=(3,3), axes=None, **kwargs):
    '''
    Parameters
    ----------
    xyz : array_like
      shape (n,3)
    '''
    n_dims = xyz.shape[1]
    if axes == None:
        print "Creating new axes."
        fig = plt.figure(figsize=(3,3), facecolor='w')
        l, r = 0.35, 0.05
        t, bm, bw = 0.05, 0.15, 0.1
        w = 1 - l - r
        h = (1 - t - 2 * bw - bm)/3.
        axes = []
        ylabels = ['x', 'y', 'z']
        for i in xrange(n_dims):
            b = 1 - t - (i + 1) * h - i * bw
            axrect = [l, b, w, h]
            ax = fig.add_axes(axrect)
            axes.append(ax)
        
        for i, ax in enumerate(axes):
            if i == 2:
                to_show = ['left', 'bottom']
            else:
                to_show = ['left']
            for loc, spine in ax.spines.iteritems():
                if loc in to_show:
                    spine.set_position(('outward', 5))
                else:
                    spine.set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            for tick in ax.xaxis.get_ticklines():
                tick.set_markeredgewidth(1.0)
            for tick in ax.yaxis.get_ticklines():
                tick.set_markeredgewidth(1.0)
            if 'bottom' in to_show:
                ax.set_xticks([0, 1.])
                ax.set_xlabel('time', labelpad=-4)
            else:
                ax.set_xticks([])
            ax.set_ylabel(ylabels[i], labelpad=-2, rotation='horizontal')
    for i, ax in enumerate(axes):
        if xs == None:
            xs = np.linspace(0, 1, xyz.shape[0])
        assert xs.shape[0] == xyz.shape[0]
        ax.plot(xs, xyz[:,i], **kwargs)
    return axes

class CorrelationPlotter(object):
    def __init__(self):
        fig = plt.figure()
        gs = GridSpec(2,2, hspace=0.3)
        self.ax_x = fig.add_subplot(gs[0,0])
        self.ax_x.set_title('X')
        self.ax_y = fig.add_subplot(gs[0,1])
        self.ax_y.set_title('Y')
        self.ax_z = fig.add_subplot(gs[1,0])
        self.ax_z.set_title('Z')
        
    def plot(self, exog, endog):
        ex = exog.T
        self.ax_x.plot(ex[0], endog, 'o')
        self.ax_y.plot(ex[1], endog, 'o')
        self.ax_z.plot(ex[2], endog, 'o')

class Plotter(object):
    color_idx = 0
    nitems = 10
    cmap = cm.jet
    color_steps = np.linspace(0,1,nitems)
    
    def __init__(self):
        fig = plt.figure()
        gs = GridSpec(2,1, height_ratios=[3,1])
        self.ax_3d = fig.add_subplot(gs[0], projection='3d')
        self.ax_2d = fig.add_subplot(gs[1])
        
    def _get_color(self):
        return self.cmap(self.color_steps[self.color_idx])        
        
    def _get_next_color(self):
        color = self._get_color()
        self.increment_color()
        return color
        
    def increment_color(self):
        self.color_idx += 1        
        
    def plot_2d(self, arr):
        color = self._get_color()
        print arr.shape
        if np.rank(arr) == 1:
            self.ax_2d.plot(arr, 'o', color=color)
        else:
            n = arr.shape[0]
            for i in xrange(n):
                self.ax_2d.plot(arr[i], 'o', color=color)
        
    def plot_3d(self, arr):
        color = self._get_next_color()
        if np.rank(arr) == 2:
            arr2 = arr.T
            self.ax_3d.plot(arr2[0], arr2[1], arr2[2], 'o', color=color)
        else:
            n = arr.shape[0]
            for i in xrange(n):
                arr2 = arr[i].transpose()
                self.ax_3d.plot(arr2[0], arr2[1], arr2[2], '-', color=color)

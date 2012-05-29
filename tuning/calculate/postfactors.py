import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# generic calc stuff
import vectors
from spherical import cart2pol
import plot_tools
from vectors import unitvec

# motorlab.tuning stuff
from motorlab.out_files import get_out_name
from motorlab.tuning.calculate.pds import bootstrap_pd_stats, calc_all_pd_uncerts
from motorlab.tuning import fit
from motorlab.tuning.calculate import factors

# plotting stuff
from spherical_stats import Lambertograph, generate_cone_circle
import split_lambert_projection
from motorlab.tuning.display import orange_plots
from mayavi import mlab

def get_pc_pd_r2(scores, bnd, preaveraged=False):
    '''Calculate r2 of pc scores to direction regression.

    Parameters
    ----------
    score : ndarray
      scores of principal components
      shape (npc, ntask, nbin)
    '''
    if preaveraged:
        pos = stats.nanmean(bnd.pos, axis=1)
        time = stats.nanmean(bnd.bin_edges, axis=1)
    else:
        rate, pos, time = bnd.get_for_regress(0, use_unbiased=False)
    ntrial, nedge, ndim = pos.shape
    nbin = nedge - 1
    npc = scores.shape[0]
    
    if (ntrial * nbin) != scores.shape[1]:
        raise ValueError("Woops! Shapes aren't compatible.")
    
    scores = scores.reshape(npc, ntrial, nbin)
    rsqs = np.zeros((npc))
    bds = np.zeros((npc, 3))
    model = 'kd'
    for i, score in enumerate(scores):
        b, cov, rsq = fit.ols_any_model(score, pos, time, model=model)
        
        bdict = fit.unpack_coefficients(b, model)
        bds[i] = bdict['d']
        rsqs[i] = rsq

    which = np.argsort(rsqs)[:-4:-1]
    dir_bds = bds[which] # get first 3 bds, sorted by rsq
    pds = unitvec(dir_bds, axis=1)
    all_angles = np.arccos(np.dot(pds, pds.T))
    idxoi = np.triu(np.ones_like(all_angles), 1).astype(bool)
    angles = all_angles[idxoi]
    return pds, rsqs, angles, which

def plot_rsq(rsq, save_dir='', n=20, color='#A02020'):
    l = np.arange(len(rsq)) + 0.6
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.bar(l[:n], rsq[:n], color=color)
    plot_tools.format_spines(ax, which=['left','bottom'])
    ax.set_xlabel('PC #', fontsize=24)
    ax.set_ylabel(r'R$^2$', fontsize=24)
    ax.set_xticks(np.arange(len(rsq)) + 1)
    ax.set_xlim([0.5,len(rsq) + 0.5])
    xtl = ax.get_xticklines()
    for xt in xtl:
        xt.set_visible(False)
    if save_dir != '':
        print "Saving"
        fig.savefig(get_out_name(save_dir, 'pca_vs_rsq', ext='pdf'), dpi=600)
    return ax

def plot_angles(pds, save_dir='', position=0):
    # align axes
    # let pds be A, B, C
    # 1) rotate C to z axis
    tpa = cart2pol(pds[2])
    pds_r0 = vectors.rotate_by_angles(pds.T, tpa[0], tpa[1], \
                                        reverse_order=True).T
    # 2) rotate B to lie in x-z plane
    # calculate angle between B[0:2] and x
    # rotate B to lie on y=0 plane
    phi = vectors.angle_between(pds_r0[1,0:2], np.array([1,0]))
    pds_r1 = vectors.rotate_by_angles(pds_r0.T, 0, phi).T
    
    # plot vectors
    ori = np.zeros_like(pds)
    fig = mlab.figure(size=(2048,2048))
    fig.scene.disable_render = True

    # draw arrows for actual vectors
    vrw = mlab.quiver3d(ori[0], ori[1], ori[2],
                        pds_r1[:,0], pds_r1[:,1], pds_r1[:,2])
    arrow_source = vrw.glyph.glyph_source.glyph_dict['arrow_source']
    vrw.glyph.glyph_source.glyph_source = arrow_source
    arrow_source.shaft_radius = 0.055
    arrow_source.shaft_resolution = 40
    arrow_source.tip_resolution = 40

    # draw really orthogonal axes as guides
    xyz = np.array([[2.,0,0],[0,2.,0],[0,0,2.]])
    # if (angle between A and y axis) > pi/2.
    # swap sign of y axis
    if np.arccos(np.dot(pds_r1[0], np.array([0,1.,0.]))) > np.pi/2.:
        xyz[1]*= -1
    vax = mlab.quiver3d(ori[0], ori[1], ori[2],
                        xyz[:,0], xyz[:,1], xyz[:,2])
    cylinder_source = vax.glyph.glyph_source.glyph_dict['cylinder_source']
    vax.glyph.glyph_source.glyph_source = cylinder_source
    cylinder_source.resolution = 40
    cylinder_source.radius = 0.03
    vax.actor.actor.mapper.scalar_visibility = False
    vax.actor.actor.property.color = (0.5,0.5,0.5)

    if position == 1:
        fig.scene.camera.position = [2.94, -3.06, 0.90]
        fig.scene.camera.focal_point = [0.45, -0.45, 0.47]
        fig.scene.camera.view_angle = 30.0
        fig.scene.camera.view_up = [-0.10, 0.076, 1.0]
        fig.scene.camera.clipping_range = [1.93, 5.78]
        fig.scene.camera.compute_view_plane_normal()

    # now draw it all
    fig.scene.disable_render = False

    if save_dir != '':
        name = get_out_name(save_dir, 'ortho_angles', ext='png')
        fig.scene.save_png(name)
    return fig.scene

def calc_tuning_trajectories(scores, bnd, preaveraged=False):
    '''Calculate trajectory of preferred direction using "kdX" model.

    Parameters
    ----------
    score : ndarray
      scores of principal components
      shape (npc, ntask * nbin)
    bnd : BinnedData
    '''
    if preaveraged:
        pos = stats.nanmean(bnd.pos, axis=1)
        time = stats.nanmean(bnd.bin_edges, axis=1)
    else:
        rate, pos, time = bnd.get_for_regress(0, use_unbiased=False)
    ntrial, nedge, ndim = pos.shape
    nbin = nedge - 1
    nscore = scores.shape[0]
    
    if (ntrial * nbin) != scores.shape[1]:
        raise ValueError("Woops! Shapes aren't compatible.")
    
    scores = scores.reshape(nscore, ntrial, nbin)
    
    pds = np.zeros((nscore, nbin, 3))
    cas = np.zeros((nscore, nbin))
    model = 'kdX'
    for i, score in enumerate(scores):
        b, cov, rsq = fit.ols_any_model(score, pos, time, model=model)
        pds[i], cas[i], md, mdse = bootstrap_pd_stats(b, cov, score, model)
        #b, bse, rsq = fit.regress_any_model(score, pos, time, model=model)
        #bd = b['d']
        #pds[i] = unitvec(bd, axis=1)
    return pds, cas
    
def plot_tuning_trajectory(pd, ca, fig=None):
    tp = cart2pol(pd)
    if fig == None:
        fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='split_lambert')    
    _tp = tp.T
    ax.plot(_tp[0], _tp[1], 'o-', color='k')
    for j, angle in enumerate(ca):
        pc_cart = generate_cone_circle(_tp[0,j], _tp[1,j],
                                       angle, resolution=240)
        pc = cart2pol(pc_cart).T
        ax.plot(pc[0], pc[1], '-', lw=0.75, color='#606060')
    return fig
    
def plot_tuning_trajectories(pd, ca, dsname='', npc=4):
    '''
    Parameters
    ----------
    pd : ndarray
      shape (nscore, nbin, 3)
    ca : ndarray
      shape (nscore, nbin), confidence angle of pd
    '''
    #tp = cart2pol(pd)
    #nbin = pd.shape[1]
    figs = []
    for i in xrange(npc):
        figs.append( plot_tuning_trajectory(pd[i], ca[i]) )
    return figs
    
def old_plot_tuning_trajectories(pd, theta, save_dir='', npc=4):
    '''
    Parameters
    ----------
    pd : ndarray
      shape (nscore, nbin - 1, 3)
    theta : ndarray
      shape (nscore, nbin)
    '''
    tp = cart2pol(pd)
    nbin = pd.shape[1]
    figs = []
    #i = 0
    for i in xrange(npc):
    #if True:
        fig = plt.figure(figsize=(10,5))
        lg = Lambertograph(n_items=pd.shape[0], fig=fig)
        for j in xrange(nbin):
            lg.plot_circle(tp[i,j,0], tp[i,j,1], theta[i,j], color='#808080')
        lg.plot_polar2(tp[i,:,0], tp[i,:,1])

        if save_dir != '':
            print "Saving"
            out_name = get_out_name(save_dir, 'pd_trajectory_pc%d' % (i+1),
                                    ext='pdf')
            fig.savefig(out_name, dpi=600)
        figs.append(fig)
    return figs
        
def plot_scores(score, ntask, nscore=8, save_dir=''):
    score_folded = factors.format_from_fa(score, ntask)
    label = ['PC %d' % (i+1) for i in range(score.shape[0])]
    fig = plt.figure(figsize=(6,4))
    orange_plots.plot_orchard(score_folded[:nscore], fig=fig,
                                    label=label, ncol=4)
    if save_dir != '':
        name = get_out_name(save_dir, 'scores_orchard', ext='pdf')
        fig.savefig(name, dpi=600)
    return fig

#def calc_pc_score_pd_rsq(scores, bnd):
#    '''Calculate r2 of pc scores to direction regression.
#
#    Parameters
#    ----------
#    score : ndarray
#      scores of principal components
#      shape (npc, ntask, nbin)
#    '''
#    pos = stats.nanmean(bnd.pos, axis=1)    
#    time = stats.nanmean(bnd.bin_edges, axis=1)    
#    ntask, nedge, ndim = pos.shape
#    nbin = nedge - 1    
#    
#    npc = scores.shape[0]
#    scores = scores.reshape(npc, ntask, nbin)
#    rsqs = np.zeros((npc))
#    bds = np.zeros((npc, 3))
#    model = 'kd'
#    for i, score in enumerate(scores):
#        b, cov, rsq = fit.ols_any_model(score, pos, time, model=model)
#                
#        bdict = fit.unpack_coefficients(b, model)
#        bds[i] = bdict['d']
#        rsqs[i] = rsq
#
#    which = np.argsort(rsqs)[:-4:-1]
#    dir_bds = bds[which] # get first 3 bds, sorted by rsq
#    pds = unitvec(dir_bds, axis=1)
#    all_angles = np.triu(np.arccos(np.dot(pds, pds.T)))
#    angles = all_angles[all_angles != 0]
#    return pds, rsqs, angles, which

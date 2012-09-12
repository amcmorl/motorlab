import numpy as np
from multiregress import multiregress
import scipy.stats as stats
from scipy.stats.distributions import norm

def unbiased_histogram(spikes, bins=None, range=None, new=None):
    '''
    Drop in replacement for np.histogram that implements the unbiased
    firing rate metric employed by Zhanwu and Rob Kass
    
    Parameters
    ----------
    spikes : array
      Array of spikes to histogram.
    bins : int or sequence
      Int for number of equal width bins, or sequence of bin-edges.
      See `np.histogram` for details.
    range : (float, float)
      Lower and upper range of bins. See `np.histogram` for more details.
    new : bool
      Whether to use new version of `np.histogram`.
      See its docstring for more detailse.
    '''
    assert type(spikes) == np.ndarray
    if np.rank(spikes) == 0:
        spikes = spikes[None]
    if range == None:
        range = (np.min(bins), np.max(bins))
        
    hist, bin_times = np.histogram(spikes,
                                   bins=bins,
                                   range=[range[0], range[1]])
    hist = hist.astype(float)
    hist += 1
    for i in xrange(hist.size):
        prespikes = spikes[spikes < bin_times[i]]
        if prespikes.size == 0:
            prespike = range[0]
        else:
            prespike = prespikes[-1]

        postspikes = spikes[spikes > bin_times[i+1]]
        if postspikes.size == 0:
            postspike = range[1]
        else:
            postspike = postspikes[0]
        time_interval = postspike - prespike
        hist[i] /= time_interval
    return hist, bin_times

def calc_pd(rates, directions, \
                with_err=False, n_samp=1000, \
                with_baseline=False):
    '''
    Calculates preferred directions of a unit from a center out task
    
    could be a method of a "unit" class
    which references the relevant data_file items

    Calculates the vector components of the preferred direction:
    c_x, c_y, and c_z from the
    b_x, b_y, and b_z partial regression co-efficients described in eqn 1 of
    Schwartz et al, 1988:

    .. math:: d(M) = b + b_x.m_x + b_y.m_y + b_z.m_z,

    and where

    .. math::

        c_x = b_x / sqrt(b_x^2 + b_y^2 + b_z^2)
        
        c_y = b_y / sqrt(b_x^2 + b_y^2 + b_z^2)
        
        c_z = b_z / sqrt(b_x^2 + b_y^2 + b_z^2)

    The algorithm regresses the firing rates (spike count / duration) against
    (1, cos theta_x, cos theta_y, cos theta_z), to give (b, b_x, b_y, b_z)

    Parameters
    ----------
    rates : array, shape (ndir,)
        1d array of firing rates in each of the (usually eight) directions
    directions : array, shape (ndir, 3)
        array of directions associated with each of the `rates`
    with_err : bool, default False
        whether to calculate and return std errors of pd components
    n_samp : integer
        when calculating error, is the number of samples from the
        theoretical distribution of PD to calculate
        
    Returns
    -------
    pd  : array, shape (3,)
      preferred direction, unit-length
    k   : scalar
      modulation depth :math:`(k = |b| = b / pd)`
    err : array, shape (3, )
        std errors of components of pds
    '''
    assert type(rates) == type(directions) == np.ndarray
    assert np.rank(rates) == 1      # (n_dirs, rate)
    assert np.rank(directions) == 2 # (ndirs, [x,y,z])
    assert directions.shape[1] == 3 # 3d co-ordinates array
    assert type(with_err) == bool
    assert type(n_samp) == int

    lengths = np.sqrt( (directions**2).sum(axis=1) )[..., None]
    cosInput = directions / lengths
    if not with_err:
        n_directions = directions.shape[0]
        xin = np.hstack((np.ones((n_directions, 1)), cosInput))
        all_b = np.linalg.lstsq(xin, rates)[0]
        b0, b = all_b[0], all_b[1:]
        k = np.sqrt(np.sum(b**2))
        # normalize to unit length
        pd = b / k
        if not with_baseline:
            return pd, k
        else:
            return pd, k, b0
    else:
        b, b_s = multiregress(cosInput, rates) # partial regression coeffs
        b = b[1:] # remove constant term for consideration of direction
        k = np.sqrt(np.sum(b**2))
        pd = b / k
        # b.shape (3,)
        return pd, b_s, k

def calc_pp(rates, positions):
    '''Calculates positional gradient for a cell, based on the nine firing
    rates that correspond to eight corner target positions and the center
    position, supplied in the positions variable

    The regression is of the equation:

    fr = b_0 + b_x * x + b_y * y + b_z * z

    and b_0, c_x, c_y, and cz are returned

    where c_x,c_y,c_z = b_x,b_y,b_z / k; k = sqrt(b_x**2 + b_y**2 + b_z**2)

    Parameters
    ----------
    rates : array_like

    positions : array_like

    Returns
    -------
    pp : array_like, shape (3,)
      vector of positional gradient
    k : scalar
      modulation depth of positional encoding
    b0 : scalar
      baseline firing rate (b0 in regression
    '''
    n_rates = rates.size
    #assert False
    all_pos = positions
    #all_pos = np.vstack((np.zeros(3), target))
    xin = np.hstack((np.ones((n_rates, 1)), all_pos))
    res = np.linalg.lstsq(xin, rates)[0]
    b0, pp = res[0], res[1:]
    k = np.sqrt(np.sum(pp**2))
    pp /= k
    return pp, k, b0

def get_task_idx(start, finish, task_list):
    '''Return index of row in target_list given by .'''
    assert type(start) == type(finish) == type(task_list) == np.ndarray
    assert start.shape == finish.shape == (3,)
    assert task_list.shape[1] == 6

    this_task = np.concatenate((start, finish), axis=1)
    diff = np.abs(task_list - this_task[None,...])
    n_same = np.sum(diff < 1e-8, axis=1)
    return np.nonzero(n_same == 6)[0].item()
#     trg_dir = task_list / np.apply_along_axis(np.linalg.norm, 1,
#                                               task_list)[..., np.newaxis]
#     vec_dir = vector / np.linalg.norm(vector)    
#     return np.nonzero( np.abs( trg_dir - vec_dir).sum(axis=1)
#                           < 1e-8)[0].item()

def smooth_PSTHs(PSTHs, smooth_sd=0.5):
    n_psths, n_bins = PSTHs.shape
    kern_x = np.linspace(0, n_bins, n_bins, endpoint=False) - n_bins/2. + 1
    #... plus one seems to remove offsets in convolution
    kern = norm.pdf(kern_x, scale=smooth_sd)
    kern /= kern.sum()
    return np.apply_along_axis(np.convolve, 1, PSTHs, kern, 'same')

def zscore(line):
    '''replacement for scipy.stats.z, but uses N-1 degrees of freedom,
    instead of N degrees of freedom'''
    return (line - np.mean(line))/stats.std(line)

def zscore_arr(PSTHs):
    return np.apply_along_axis(zscore, 1, PSTHs)

def twopt_interp(x0, x1, y0, y1, yk):
    m = (y1 - y0) / (x1 - x0)
    c = y1 - m * x1
    return (yk - c) / m

# def fractional_histogram(spikes, bins=None, range=None, new=None):
#     '''a drop in replacement for np.histogram that implements the unbiased
#     firing rate metric employed by Zhanwu and Rob Kass'''
#     bin_edges = np.linspace(range[0], range[1], bins + 1, endpoint=True)
#     for i in xrange(bins + 1):
#         bin_start = bin_edges[i]
#         bin_stop = bin_edges[i+1]
#     return hist, bin_edges

# def create_dir_axes(lmargin, bmargin, width, height, space, fig):
#     dirs = list(util.target)
#     axs = []
#     for dr in dirs:
#         dr_offset = (dr == 1).astype(int)
#         left = lmargin \
#                + dr_offset[0] * (width + space) \
#                + dr_offset[2] * 2 * (width + space)
#         bottom = bmargin + dr_offset[1] * (height + space)
#         axes_rect = [left, bottom, width, height]
#         axs.append(fig.add_axes(axes_rect))
#     return axs

# target = np.array([[-1, -1, -1], \
#                    [-1, -1,  1], \
#                    [-1,  1, -1], \
#                    [-1,  1,  1], \
#                    [ 1, -1, -1], \
#                    [ 1, -1,  1], \
#                    [ 1,  1, -1], \
#                    [ 1,  1,  1]])


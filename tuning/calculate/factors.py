import numpy as np
from scipy import stats
import mdp
from scikits.learn.decomposition.pca import PCA
from scikits.learn.decomposition.fastica_ import FastICA
from numpy.testing import assert_array_almost_equal

# py-to-r stuff
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
_varimax = ro.r('varimax')

def get_reduced_neurons(score, which_scores, weight, bnd, preaveraged=False):
    '''
    Project firing activity (in `bnd`) on to a space of reduced factors, 
    given by `which_scores` of `scores`.
    
    Parameters
    ----------
    score : ndarray
      (npc, nobs) array of factors identified by factors.calc_pcs_mdp
    which_scores : ndarray
      indices of scores to keep
    weight : ndarray
      (npc, nvar) array of eigenvector values
    bnd : BinnedData
      contains neural data to project (uses rates)
    preaveraged : bool
      dictates how to reformat data
    
    Returns
    -------
    rates : ndarray
    '''
    assert score.shape[0] == weight.shape[0] # npc
    
    # select subset of scores
    reduced_score = score[which_scores]
    reduced_weight = weight[which_scores]

    # this sxn needs to get from
    #     (ntask, nrep, nunit, nbin) to (nunit, ntask * nbin)
    data = format_for_fa(bnd, preaverage=preaveraged).T
    
    # project_scores_to_var_space expects no repeats (i.e. preaveraged)
    # uses `data` to get mean values
    reduced_neurons_flat = project_scores_to_var_space( \
        reduced_score, reduced_weight, data)
                                                       
    # returns data in shape (nscore|nunit, ntask, nrep, nbin)
    return format_from_fa(reduced_neurons_flat, bnd, \
        preaveraged=preaveraged)

def varimax(weight):
    '''
    Perform varimax rotation of weights matrix.

    Uses r varimax call. Varimax rotates weights matrix to consolidate
    weight into fewest possible units for each score.
    
    Parameters
    ----------
    weight : ndarray
      unrotated weight matrix, shape = (npc, nvar)
    
    Returns
    -------
    rot_weight : ndarray
      rotated weight matrix
    rot_mat : ndarray
      rotation matrix
    '''
    out = _varimax(weight)
    rot_weight, rot_mat = [np.array(x) for x in out]
    return rot_weight, rot_mat

def format_for_fa(bnd, preaverage=False, use_unbiased=False):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
    '''
    if preaverage:    
        # average across repeats
        if not use_unbiased:
            rate = stats.nanmean(bnd.get_rates(), axis=1)
        else:
            rate = stats.nanmean(bnd.unbiased_rate, axis=1)
    else:
        if not use_unbiased:
            rate = bnd.get_rates_flat(with_nans=False)
        else:
            rate = bnd.get_unbiased_rate_flat(with_nans=False)

    ntask, nunit, nbin = rate.shape
    rate = np.transpose(rate, [1, 0, 2])
    
    # gives nunit, ntask, nbin
    rate = np.reshape(rate, [nunit, ntask * nbin])
    # gives nunit, ntask * nbin = dimensions, observations

    # mdp and scikits.learn.pca take data in form (observations, dimensions)
    return rate.T

def format_from_fa(score, bnd, preaveraged=False):
    '''
    Parameters
    ----------
    score : ndarray
      scores from factor analysis,
      in unfolded form, i.e. (nscore, ntask * nrep * nbin)
    bnd : BinnedData
      original data, so that shape and nans can be taken into account
    preaveraged : bool
      did we preaverage repeats before calculating scores
      
    Returns
    -------
    data : ndarray
      formatted into shape (nscore, ntask, [nrep,] nbin)
      where presence of nrep depends on `preaveraged`
    '''
    ntask, nrep, nunit, nbin = bnd.count.shape    
    nscore = score.shape[0]

    if not preaveraged:
        valid_trials = bnd.get_notnans()
        valid_bins = np.repeat(valid_trials, nbin)
    else:
        nrep = 1
        valid_bins = slice(None,None)

    # make receptacle
    formatted = np.zeros([nscore, ntask * nrep * nbin]) + np.nan        
    formatted[:,valid_bins] = score
    
    # now reverse formatting
    if preaveraged:
        reshaped = np.reshape(formatted, (nscore, ntask, nbin))
    else:
        reshaped = np.reshape(formatted, (nscore, ntask, nrep, nbin))
    return reshaped

def calc_pca(bnd, npc=None, preaverage=False, use_unbiased=False, \
    method='mdp'):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
    npc : int or None, optional
      number of PCs to calculate, defaults to None
    preaverage : bool
      average across repeats?
      
    Returns
    -------
    score : ndarray
      (npc, nobs)
    weight : ndarray
      (npc, nvar)
    '''
    assert method in ['mdp', 'skl']
    data = format_for_fa(bnd, preaverage=preaverage,
                         use_unbiased=use_unbiased)
    if method == 'mdp':    
        pca_node = mdp.nodes.PCANode(output_dim=npc)
        score = pca_node.execute(data)
        weight = pca_node.get_projmatrix()
    elif method == 'skl':
        pca_obj = PCA(n_components=npc)
        score = pca_obj.fit(data).transform(data)
        weight = pca_obj.components_.T
    return score.T, weight.T

def calc_pcs_mdp(bnd, npc=None, preaverage=False, use_unbiased=False):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
    npc : int or None, optional
      number of PCs to calculate, defaults to None
    preaverage : bool
      average across repeats?
    '''
    print "calc_pcs_mdp deprecated! Use calc_pca instead."
    return calc_pca(bnd, npc=npc, preaverage=preaverage, 
        use_unbiased=use_unbiased, method='mdp')
    
def calc_pcs_learn(bnd, npc=None, preaverage=False, use_unbiased=False):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
    npc : int or None, optional
      number of PCs to calculate, defaults to None
    '''
    print "calc_pcs_learn deprecated! Use calc_pca instead."
    return calc_pca(bnd, npc=npc, preaverage=preaverage,
        use_unbiased=use_unbiased, method='skl')

def calc_pcs_variance_explained_mdp(bnd, preaverage=False, 
                                   use_unbiased=False):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
    preaverage : bool
      average across repeats?
    use_unbiased : False
      use the unbiased spike rates calculated using Rob Kass's
      spike rate method
    '''
    data = format_for_fa(bnd, preaverage=preaverage,
                         use_unbiased=use_unbiased)
    pca_node = mdp.nodes.PCANode()
    # have to run execute to calculate attributes
    score = pca_node.execute(data)
    eigvals = pca_node.d
    var_explained = eigvals / eigvals.sum()
    return var_explained
    
def _check_pca(data, weight, score):
    norm_data = data - stats.nanmean(data)
    assert_array_almost_equal(np.dot(norm_data, weight), score)

def project(data, weight):
    '''
    Parameters
    ----------
    data : ndarray
      shape (nobs, nvar)
    weight : ndarray
      shape (npc, nvar)
      
    Returns
    -------
    score : ndarray
      shape (npc, nobs)
    '''
    assert np.rank(data) == np.rank(weight) == 2
    assert data.shape[1] == weight.shape[1]
    
    # average across observations, 0th dimension
    norm_data = data - stats.nanmean(data)
    
    # (nobs, nvar) x (nvar, npc) -> (nobs, npc)
    score = np.dot(norm_data, weight.T)
    return score.T  # -> (npc, nobs)  
    
def project_scores_to_var_space(score, weight, data):
    '''
    Project reduced scores, via reduced weights, up to neuron space    
    
    Parameters
    ----------
    score : ndarray
      shape (npc, nobs), i.e. (nscore, ntask [* nrep] * nbin)
    weight : ndarray
      shape (npc, nvar), i.e. (nscore, nunit)
    data : ndarray
      shape (nvar, nobs), data from which to get mean
      
    Returns
    -------
    projected : ndarray
      shape (nvar, nobs), i.e. (nunit, ntask * nbin)
    '''
    assert np.rank(score) == np.rank(weight) == np.rank(data) == 2
    assert score.shape[0] == weight.shape[0] # npc
    assert score.shape[1] == data.shape[1]   # nobs
    assert weight.shape[1] == data.shape[0]  # nvar
    
    # take average over observations
    mean = stats.nanmean(data, axis=1)
    return (np.dot(weight.T, score) + mean[:,None])
    
# scikits.learn ###############################################################

def calc_ics_learn(bnd, npc=None):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
    npc : int or None, optional
      number of PCs to calculate, defaults to None
    '''
    data = format_for_fa(bnd)
    ica_obj = FastICA(n_comp=npc)
    source = ica_obj.fit(data).transform(data)
    weight = ica_obj.get_mixing_matrix()
    return source.T, weight.T

# other MDP ###################################################################

def calc_pcs_mdp_preformatted(data, npc=None, use_unbiased=False):
    '''
    Parameters
    ----------
    data : ndarray
      (nvar, nobs) array to do PCA on
    npc : int or None, optional
      number of PCs to calculate, defaults to None
    '''
    assert np.rank(data) == 2
    pca_node = mdp.nodes.PCANode(output_dim=npc)
    score = pca_node.execute(data)
    weight = pca_node.get_projmatrix()
    return score.T, weight.T

def calc_ics_mdp(bnd):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
    '''
    data = format_for_fa(bnd)
    ica_node = mdp.nodes.FastICANode()
    score = ica_node.execute(data)
    weight = ica_node.filters.T
    return score.T, weight.T
    
def calc_fa(bnd, nfa=None):
    '''
    Parameters
    ----------
    bnd : BinnedData
      binned data
      
    Notes
    -----
    uses mdp
    '''
    data = format_for_fa(bnd)
    fa_node = mdp.nodes.FANode(output_dim=nfa)
    score = fa_node.execute(data)
    #print score.flatten()[10]
    return score.T

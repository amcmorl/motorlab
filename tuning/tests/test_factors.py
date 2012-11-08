from motorlab.tuning.calculate.factors import \
    project_scores_to_var_space, project
import numpy as np
from sklearn.decomposition.pca import PCA

def test_project_scores_to_var_space():
    '''
    '''
    nunit, ntask, nrep, nbin = 4,3,2,5
    data = np.random.normal(10., 2, size=(nunit, ntask, nrep, nbin))
    
    # data needs to be in form (observations, dimensions)
    data_for_fa = np.mean(data, axis=2).reshape(nunit, ntask * nbin)

    def _calc_factors(data, npc=None):
        pca_obj = PCA(n_components=npc)
        score = pca_obj.fit(data).transform(data)
        # transpose here makes the output match with mdp
        weight = pca_obj.components_.T
        return score.T, weight.T

    score, weight = _calc_factors(data_for_fa.T)

    # case 1: weights and scores from averaged data
    out = project_scores_to_var_space(score, weight, data_for_fa)
    np.testing.assert_array_almost_equal(out, data_for_fa)

    # case 2: weights from averaged data, scores from unaveraged data
    unaveraged = np.reshape(data, (nunit, ntask * nrep * nbin))
    unav_score = project(unaveraged.T, weight)
    out = project_scores_to_var_space(unav_score, weight, unaveraged)
    np.testing.assert_array_almost_equal(out, unaveraged)
    
    

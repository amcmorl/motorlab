'''
Test fit.py routines.
'''

from motorlab.tuning.fit import glm_any_model

import os
import sys
paper_dir = os.environ['DBOX'] + '/papers/directionality_paper'
proj_dir  = paper_dir + '/code'
sim_dir   = paper_dir + '/figures/fig_ctrl_sim'

if proj_dir not in sys.path:
    sys.path.append(proj_dir)

import tuning_project as tp
from simulate_cells import ChangingPDPopulation, ConstantPDPopulation
from motorlab.binned_data import load_binned_data

def test_glm_any_model():
    # make a simulated neuron with a changing PD
    pars = tp.parameters['std']
    bnd_file = pars.bnd_file % tp.ds_frank
    real_bnd = load_binned_data(bnd_file)

    changing_pd_pars = ParameterSet()
    changing_pd_pars.b0_mean = 20.
    changing_pd_pars.b0_sd   =  5.
    changing_pd_pars.md_mean = 20.
    changing_pd_pars.md_sd   = 10.
    nbin = 10

    mds = np.random.normal(pars.md_mean, pars.md_sd, size=ncell * 2)
    pds = uniform_rvs_cart(size=ncell * 2)
    BDs = pds * mds[:,None]
    pdas, pdbs = BDs[:ncell], BDs[ncell:]
    
    cell = ChangingPDCell(B, nbin=nbin)
    # try to recover that PD using glm_any_model

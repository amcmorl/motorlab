import numpy as np
import motorlab.tuning.datacollection as datac
from motorlab.load import get_file_list, get_units_list, pick_units
import motorlab.tuning.sort as sort

class Options:
    files = get_file_list()
    unit_names = pick_units(get_units_list(), '>100')[0:1]
    units = [(x, 0.1) for x in unit_names]
    nbin = 80                # number of bins
    align = '323 simple'     # alignment scheme

def test_get_sort_by_pd_co():
    options = Options()
    dc = datac.DataCollection(options.files)
    dc.add_units(options.units)
    bnd = dc.construct_PSTHs(nbins=options.nbin, align=options.align)
    cobypd = sort.get_sort_by_pd_co(bnd)
    tasks = bnd.parent.tasks
    dirs = tasks[:,3:] - tasks[:,:3]
    np.testing.assert_almost_equal(dirs[cobypd][:,:26], dirs[cobypd][:,26:])

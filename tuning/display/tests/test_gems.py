import numpy as np
import matplotlib.pyplot as plt
from motorlab.tuning.display import gems
from motorlab.datasets import datasets
from motorlab.tuning.datacollection import DataCollection

def get_dc(units=None):
    ds = datasets['small']
    dc = DataCollection(ds.get_files())
    if units:
        dc.add_units([(x, 0.1) for x in units])
    return dc

def test_pcolor_targets():
    dc = get_dc()
    tasks = dc.tasks
    co = np.all(tasks[:,:3] == 0., axis=1)
    cot = tasks[co]
    co_dir = cot[:,:3] - cot[:,3:]
    
    data = co_dir[:,0]
    gems.pcolor_targets(data)

def test_pcolor_series():
    dc = get_dc(['Unit002_1'])
    bnd = dc.make_binned(nbin=10)
    co = np.all(dc.tasks[:,:3] == 0., axis=1)
    data = np.mean(bnd.PSTHs, axis=1)[co,0]
      # take avg of repeats, co, & 1st unit
    fig = gems.pcolor_target_series(data, order='co')
    return fig
    
if __name__ == "__main__":
    test_pcolor_series()
    plt.show()

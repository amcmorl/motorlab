from motorlab.tuning_change.simulate import open_loop as ol
from numpy.testing import assert_, assert_equal

def test_make_simple_cart_coords():
    task, pos = make_simple_cart_coords(20)
    # check shapes are consistent
    ntask = task.shape[0]
    ndim = task.shape[1] / 2
    assert_equal(pos.shape, (ntask, 20, ndim))

def test_format_as_inputs():
    task, pos = make_simple_cart_coords(20)
    ntask = task.shape[0]
    ndim = pos.shape[-1]
    npt = pos.shape[1]
    inputs = format_as_inputs(task, pos)
    assert_equal(inputs.shape, (ntask, npt, ndim * 2))

def test_make_cPDs():
    task, pos = ol.make_simple_cart_coords(20)
    nout = 2
    out = ol.make_cPDs(task[:,3:], pos, nout)
    ntask, npt = pos.shape[0:2]
    assert_equal(out.shape, (ntask, npt, nout))
    
def test_make_cart_network():
    pos, out, tgt = ol.make_cart_network(nout=10, cpd_frac=0.3)
    

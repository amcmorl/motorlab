import numpy as np
import motorlab.tuning_change.datacollection as d_c
from tempfile import TemporaryFile

def test_save_load():
    PSTHs = np.tile(np.arange(10, dtype=float)[None,None,None,:], (8, 2, 1, 1))
    bin_edges = np.tile(np.arange(11)[None,None,:], (8, 2, 1))
    pos = np.tile(np.arange(11)[None,None,:,None], (8, 2, 1, 3))
    align_start_bins = np.tile(np.arange(8)[:,None], (1, 2))
    align_end_bins = np.tile(np.arange(8)[:,None], (1, 2))    
    srt = d_c.SortedData(None, PSTHs, bin_edges, pos,
                         align_start_bins, align_end_bins)
    outfile = TemporaryFile()
    srt.save(outfile)
    outfile.seek(0)
    srtn = d_c.load_sorted_data(outfile)
    np.testing.assert_array_almost_equal(srt.PSTHs, srtn.PSTHs)
    np.testing.assert_array_almost_equal(srt.bin_edges, srtn.bin_edges)
    np.testing.assert_array_almost_equal(srt.pos, srtn.pos)
    np.testing.assert_array_almost_equal(srt.align_start_bins, srtn.align_start_bins)
    

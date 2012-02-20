import numpy as np

'''
fractional interval calculator 
 - see docstring for get_fractional_interval
'''

def test_get_fractional_interval():
    '''
    test suite for fractional interval calculator
    '''
    # normal case: spikes flank (precede and follow all bin starts
    # and ends
    bin_times = np.array([1, 2, 3, 4])
    spike_times = np.array([0.5, 0.8, 1.4, 1.9, 3.7, 4.4, 5.4])
    answer = np.array([(1.4 - 1) / (1.4 - 0.8) + 1 + 0.1 / 1.8, # 1-2s
                       1 / 1.8, # 2-3s
                       0.7 / 1.8 + 0.3 / 0.7]) # 3-4s
    result = get_fractional_interval(bin_times, spike_times)
    np.testing.assert_allclose(answer, result) # 1
    
    # case 2: no spike before first bin
    bin_times = np.array([1, 2, 3, 4])
    spike_times = np.array([1.4, 1.9, 3.7, 4.4, 5.4])
    answer = np.array([1 + 0.1 / 1.8, # 1-2s
                       1 / 1.8, # 2-3s
                       0.7 / 1.8 + 0.3 / 0.7]) # 3-4s
    result = get_fractional_interval(bin_times, spike_times)
    np.testing.assert_allclose(answer, result) # 2
    
    # case 3: no spike in first bin
    bin_times = np.array([1, 2, 3, 4])
    spike_times = np.array([2.1, 3.7, 4.4, 5.4])
    answer = np.array([0, # 1-2s
                       0.9 / 1.6, # 2-3s
                       0.7 / 1.6 + 0.3 / 0.7]) # 3-4s
    result = get_fractional_interval(bin_times, spike_times)
    np.testing.assert_allclose(answer, result) # 3
    
    # case 4: no spike after last bin
    bin_times = np.array([1, 2, 3, 4])
    spike_times = np.array([0.5, 0.8, 1.4, 1.9, 3.7])
    answer = np.array([(1.4 - 1) / (1.4 - 0.8) + 1 + 0.1 / 1.8, # 1-2s
                       1 / 1.8, # 2-3s
                       0.7 / 1.8]) # 3-4s
    result = get_fractional_interval(bin_times, spike_times)
    np.testing.assert_allclose(answer, result) # 4
    
    # case 5: no spike in last bin
    bin_times = np.array([1, 2, 3, 4])
    spike_times = np.array([0.5, 0.8, 1.4, 1.9])
    answer = np.array([(1.4 - 1) / (1.4 - 0.8) + 1, # 1-2s
                       0, # 2-3s
                       0]) # 3-4s
    result = get_fractional_interval(bin_times, spike_times)
    np.testing.assert_allclose(answer, result) # 5
    
def do_one_fractional_interval():
    bin_times = np.linspace(0.1,0.5,100)
    spike_times = np.random.uniform(0, 0.6, size=50)
    spike_times.sort()
    answer = np.array([(1.4 - 1) / (1.4 - 0.8) + 1 + 0.1 / 1.8, # 1-2s
                       1 / 1.8, # 2-3s
                       0.7 / 1.8 + 0.3 / 0.7]) # 3-4s
    get_fractional_interval(bin_times, spike_times)
    
def get_fractional_interval(bin, spike):
    '''
    Calculate the fractional interval histogram of `spikes` in `bins`.
    
    Parameters
    ----------
    bin : array_like, sequence
      vector of bin times, including end time of last bin,
      must be sorted in ascending order
    spike : array_like, sequence
      vector of spike times, must be sorted in ascending order
    
    Notes
    -----
    Fractional interval sums the number of fractional intervals in each
    bin, so that if a bin is between 1 and 2 s, and there are spikes at
    0.9, 1.5, 1.9, and 2.3 s, then the fractional interval count for
    that bin is 0.5 / 0.6 + 1 + 0.1 / 0.4 = 2.0833
    '''
    cur_spike = 0
    cur_bin = 0
    
    nspike = len(spike)
    nbin = len(bin)
    
    fcount = np.zeros(nbin - 1)
    while cur_bin < nbin - 1:
        
        # normal case: first spike precedes first bin start
        # loop to find first spike in bin
        while spike[cur_spike] < bin[cur_bin]:
            cur_spike += 1

        if cur_spike != 0:
            # corner case: first spike is later than first bin start
            # -> ignore preceding fraction then
            fore_interval = spike[cur_spike] - spike[cur_spike - 1]
            fore_in_bin_interval = spike[cur_spike] - bin[cur_bin]
            fcount[cur_bin] += fore_in_bin_interval / fore_interval
        
        # spike intervals fully inside bin
        while spike[cur_spike] < bin[cur_bin + 1]:
            fcount[cur_bin] += 1
            cur_spike += 1
            if cur_spike == nspike:
                # corner case: last spike is earlier than last bin
                # -> ignore following fraction then
                # putting in while loop avoids index
                # spike[cur_spike] when `cur_spike` is out of range
                # nothing more to do, so tidy up and return
                fcount[cur_bin] -= 1
                return fcount
        # finish with cur_spike being after bin end
        
        if cur_spike != 0:
            # if cur_spike was still 0 => 
            # no spikes in or preceding this bin => should be 0
            fcount[cur_bin] -= 1 # for last fractional interval
        
            # normal case: last spike follows last bin
            aft_interval = spike[cur_spike] - spike[cur_spike - 1]
            aft_in_bin_interval = bin[cur_bin + 1] - spike[cur_spike - 1]
            fcount[cur_bin] += aft_in_bin_interval / aft_interval
        
        cur_bin += 1
    return fcount

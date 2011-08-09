import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

default_cables = {'1':1, '2':2, '3':3}

# start with plx channels
def _calc_value_for_chan(grading_list, method='sum'):
    '''
    Convert grading codes into values for each unit.

    The conversion is arbitrarily s=10, g=8, f=5, p=2.

    Parameters
    ----------
    grading_list : list
      list of grade codes for that channel (generally one per unit)
    method : string
      method to handle multiple units - currently 'max' or 'sum'

    Returns
    -------
    value : scalar
    '''
    values = {'s': 10, 'g' : 8, 'f': 5, 'p' : 2}
    values_list = [values[k] for k in grading_list]
    if method == 'sum':
        value = sum(values_list)
    elif method == 'max':
        value = max(values_list)
    return value


def get_chans_values_lists(chans):
    '''
    Construct two lists, each of equal length, for channel numbers and corresponding values.

    Parameters
    ----------
    chans : dict
      channel numbers (str(int)) as keys, and list of unit codes as value

    Returns
    -------
    chans_list : list, len n
      channel numbers, int
    values : list, len n
      values for each channel, int
    '''
    chans_list = chans.keys()
    values_list = [_calc_value_for_chan(chans[y]) for y in chans_list]
    chans_list = [int(x) for x in chans_list]
    return chans_list, values_list


# remap plexon channels to headstage channels
def _remap_plx_to_headstages(plx_chan, cables=default_cables, verbose=False):
    '''
    Convert plexon channel numbers to headstage channel numbers, 
    which takes care of cable setup between plexon box and headstage.

    Parameters
    ----------
    plx_chan : int
      channel number within plexon system
    cables : dict
      keys are (str(int)) bank number on Plexon box and pre-amps
      value (int) is bank number on CerePort connector
      default straight-thru all channels
    verbose : bool
      print mapping?

    Returns
    -------
    hd_chan : int
      channel number in headstages
    '''
    bank_size = 32
    little_num = ((plx_chan - 1) % bank_size) + 1
    plx_bank = (plx_chan - 1) / bank_size + 1
    try:
        hd_bank = cables[str(plx_bank)]
    except KeyError:
        raise ValueError, "Bank %d was apparently not plugged in.\n" \
            "Supplied cable mapping was: %s" % (plx_bank, str(cables))
    hd_chan = bank_size * (hd_bank - 1) + little_num
    if verbose:
        print "Plexon channel %d (%d of bank %d) -> " \
            "headstage channel %d (bank %d)" % \
            (plx_chan, little_num, plx_bank, hd_chan, hd_bank)
    return hd_chan


def _convert_hd_chan_to_array_pos(chan, array_map):
    '''Converts a channel number to the corresponding position on the array.

    Parameters
    ----------
    chan : scalar 
     channel number as seen by Plexon Box
    array : string
      file name of array-wire mapping, which is a txt file: chan x y
    cables : dictionary
      cable mapping, default straight-thru all channels
      key is bank number on Plexon box and pre-amps
      value is bank number on CerePort connector

    Returns
    -------
    pos : tuple
      position of electrode tine on array
    '''
    assert chan < 100
    key = "%02d" % (chan)
    return array_map[key]


# exposed API
def display_array(plx_chans, array_map, cables=default_cables, values=None,
                  cmap=mpl.cm.gray, verbose=False, vmin=None, vmax=None):
    '''
    Draw an image of the array with channels coded according to values
    
    Parameters
    ----------
    plx_chans : list
      channel numbers in plexon system
    array_map : dict
      keys are str(int) of headstage channels
      values are tuple of (row, col) of channel on array
    cables : dictionary
      cable mapping, default straight-thru all channels
      key is bank number on Plexon box and pre-amps
      value is bank number on CerePort connector
    values : list, default None
      values to code channel with
      if None, then ones are used to indicate position of channels
    cmap : matplotlib color map
      color map to use for array image
    verbose : bool, default False
      print mappings?
    
    Returns
    -------
    axes : matplotlib axes object
      axes containing array image
    
    Notes
    -----
    Wire is to the right of the image.
    '''
    #hd_chans = [remap_plx_to_headstages(x) for x in plx_chans]
    disp_arr = np.zeros((10, 10), dtype=int)
    for i_chan, plx_chan in enumerate(plx_chans):
        hd_chan = _remap_plx_to_headstages(plx_chan, cables=cables)
        ar_x, ar_y = _convert_hd_chan_to_array_pos(hd_chan, array_map)
        if verbose:
            print "Plexon %02d -> Headstage %02d -> Array %02d, %02d" % \
                (plx_chan, hd_chan, ar_x, ar_y)
        if values != None:
            disp_arr[ar_x, ar_y] = values[i_chan]
        else:
            disp_arr[ar_x, ar_y] = 1
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.02,0.00,0.96,1.])
    if vmin == None:
        vmin = np.min(disp_arr)
    if vmax == None:
        vmax = np.max(disp_arr)
    ax.imshow(disp_arr, interpolation='nearest', cmap=cmap,
              vmin=vmin, vmax=vmax, origin='lower')
    ax.text(1.005,0.5, '|', fontsize='small',
            transform=ax.transAxes, rotation='vertical')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    return ax

    
def _convert_elec_to_row_col(elec):
    '''
    Used in load_array_map to convert electrode numbers to 0-based index
    grid positions.
    
    Notes
    -----
    array map per Blackrock is:
    
     1  2  3  4  5  6  7  8  9  10
    11 12...
    
    i.e. zeros are last, not first
    '''
    # make 0 first
    elec -= 1
    row = elec / 10
    col = elec % 10
    return row, col


def load_array_map(filename):
    '''Loads an array map from a file.

    Parameters
    ----------
    filename : str
      path to file to read

    Returns
    -------
    array_map : dict
      keys are str(int) of headstage channels
      values are tuple of (row, col) of channel on array
    '''
    f = open(filename)
    lines = f.readlines()
    array_map = {}
    for line in lines:
        if not line[0] == '#':
            # ignore lines starting with # as comments
            wire, elec = [int(x) for x in line.split()]
            array_map["%02d" % (wire,)] = _convert_elec_to_row_col(elec)
    return array_map



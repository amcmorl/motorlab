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
    Convert plexon channel numbers to headstage channel numbers, which takes care of cable setup between plexon box and headstage.

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
                  cmap=mpl.cm.gray, verbose=False):
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
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(disp_arr, interpolation='nearest', cmap=cmap, vmin=0, vmax=20)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    return ax

    
def _convert_elec_to_row_col(elec):
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
        wire, elec = [int(x) for x in line.split()]
        array_map["%02d" % (wire,)] = _convert_elec_to_row_col(elec)
    return array_map


def _is_comment(line):
    '''
    Comments are lines starting (maybe after whitespace) with #.
    '''
    stripped = line.strip()
    if len(stripped) > 0:
        if stripped[0] == '#':
            return True
        else:
            return False


def _load_unit_grading_block(lines):
    '''Loads one block of unit grading codes and channels from a units file.

    Parameters
    ----------
    lines : list of lines to process

    Returns
    -------
    chans : dict
      keys are str(int) channel numbers
      values are list of str unit grading codes
      
    Notes
    -----
    Block starts at first line given, and continues until a commented line or
    all lines are used. Blank lines will cause an error.
    '''
    chans = {}
    if len(lines) > 0:
        okay = True
    else:
        okay = False
    while okay:
        line = lines.pop(0)
        if not _is_comment(line):
            bits = line.split()
            chan = int(bits[0])
            units = bits[1:]
            chans['%02d' % chan] = units
        else:
            okay = False
        if len(lines) == 0:
            okay = False
    return chans


def _skip_unit_grading_block(lines):
    '''
    Works inplace to remove leading lines up to and including next comment.
    '''
    if len(lines) > 0:
        okay = True
    else:
        okay = False
    while okay:
        line = lines.pop(0)
        if _is_comment(line):
            okay = False
        if len(lines) == 0:
            okay = False
    return 


def load_unit_grading(filename, area):
    '''
    Load a unit grading file.

    Parameters
    ----------
    filename : str
      path to file to open
    area : str
      name of area to load from file
      this value is compared with line before block to read
      typical values are 'ANT' or 'POST'

    Returns
    -------
    chans : dict
      keys are str(int) channel numbers
      values are list of str unit grading codes

    Notes
    -----
    File format is: first line should be a comment (starts with #) containing
    name of area listed next (ANT or POST).
    Channel listing is one line per channel, each line has format:
      channel_number  unit_code  unit_code
    Next block is indicated by another commented line ane name of next area.
    Blank lines will cause errors.
    '''
    assert type(filename) == type(area) == str
    f = open(filename)
    lines = f.readlines()
    line = lines.pop(0)
    if _is_comment(line):
        if area in line:
            chans = _load_unit_grading_block(lines)
        else:
            _skip_unit_grading_block(lines)
            chans = _load_unit_grading_block(lines)
    else:
        raise ValueError, "Wrong file format: first line should be a comment."
    return chans


def count_units(chans):
    '''
    Count number of units in channels dictionary.

    Parameters
    ----------
    chans : dict
      keys are str(int)? channel number
      values are list of str unit codings
      
    Returns
    -------
    count : int
      number of units
    '''
    count = 0
    for units in chans.itervalues():
        count += len(units)
    return count
        
# main
if __name__ == "__main__":
    example()

def example():
    '''
    Example code for plotting an array image with unit codings.
    '''
    fdir = '/home/amcmorl/files/pitt/frank/'
    units_file = fdir + 'data/Units060809_ANT.txt'
    units = load_unit_grading(units_file, 'ANT')
    chans_list, values_list = get_chans_values_lists(units)

    reverse_cables = {'1':3, '3':1}
    #array_file = fdir + 'frank_array_map_2084-2SN0366.txt'
    array_file = fdir + 'frank_array_map_1640-9SN0368_ant.txt'
    array_map = load_array_map(array_file)
    axis = display_array(chans_list, array_map, cables=reverse_cables,
                         values=values_list)

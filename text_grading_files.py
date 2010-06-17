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

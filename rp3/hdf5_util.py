import h5py

# create a field name from a list of the fieldnames at each level
fjoin = lambda l: '/'.join(l)

def iter_level(file, fields, parent):
    for f in fields:
        print "Checking %s" % (f)
        if type(file[f]) == h5py.highlevel.Group:
            new_parent = parent + '/' + f
            iter_level(file, file[f].keys(), new_parent)

def print_file_hierarchy(file):
    '''
    Print the entire key hierarchy of an hdf5 file.
    '''
    if not h5py.is_hdf5(file):
        raise ValueError('sqrrp! needs to be an hdf5 file')

    f = h5py.File(file)
    iter_level(f, file.keys(), '')

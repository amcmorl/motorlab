import os.path
import h5py
import scipy.io as io
import types
import sys
import numpy as np

def get_children(parent):
    '''
    return a list of the names of all children objects of `parent`
    '''
    # cheat if we can
    if hasattr(parent, '_fieldnames'):
        children = parent._fieldnames
        children.sort()
        return children
    elif hasattr(parent, '__dict__'):
        children = parent.__dict__.keys()
        children.sort()
        return children
    else:
        return None

def indent_print_name(name, level, obj):
    '''
    print the name, and if relevant, shape and dtype of each object,
    indented by the level
    '''
    indent = 4
    sys.stdout.write(" " * level * indent + name)
    obj_type = type(obj)
    #sys.stdout.write(str(obj_type))
    if obj_type == np.ndarray:
        sys.stdout.write(" " + str(obj.shape) + ' ' + str(obj.dtype))
    sys.stdout.write('\n')

def walk_children(parent, level=0, cb=None):
    '''recursively walk down tree of data structure, optionally
    calling `cb` at each instance
    
    `cb` should be a function which takes the signature:
      name, level, obj
    '''
    children = get_children(parent)
    if type(children) != type(None):
        for child_name in children:
            child = getattr(parent, child_name)
            if cb != None:
                cb(child_name, level, child)
            walk_children(child, level=level+1, cb=cb)

def get_mat_field(self, *field):
    '''
    field loader for matlab pre v7.3 format files that accepts same
    signature as hdf5_field_loader

    Parameters
    ----------
    f : dict
      loaded matlab file
    field : list of string
      data tree path to field
    '''
    field = list(field)
    current = self.f[field.pop(0)]
    for layer in field:
        current = current.__dict__[layer]
    return current

def get_hdf5_field(self, *field):
    '''
    Parameters
    ----------
    f : h5py.highlevel.File
      loaded hdf5 file
    field : list of string
      data tree path to field
    '''
    fieldname = '/'.join(field)
    return self.f[fieldname]

class FormattedData(object):
    '''
    generic formatted data file wrapper

    Attributes
    ----------
    fname : string
      associated file name
    format : string
      either 'hdf5' or 'mat'

    Methods
    -------
    get_field -

    Parameters
    ----------
    fname : string
      path to file
    '''
    def __init__(self, fname):
        if not os.path.exists(fname):
            raise ValueError('%s does not exist' % (fname))
        else:
            self.fname = fname
        if h5py.is_hdf5(fname):
            self.format = 'hdf5'
            self.f = h5py.File(fname)
            self.get_field = types.MethodType(get_hdf5_field, self)
        else:
            self.format = 'mat'
            self.f = io.loadmat(fname, struct_as_record=False,
                                squeeze_me=True)
            self.get_field = types.MethodType(get_mat_field, self)
            
    def list_fields_from(self, *field_name):
        '''
        list field tree structure
        '''
        field = self.get_field(*field_name)
        walk_children(field, cb=indent_print_name)

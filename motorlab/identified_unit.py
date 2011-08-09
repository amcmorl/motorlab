from motorlab.data_files import CenOut_VR_RTMA_10_File, versions, \
    CenOut_3d_data_21_File
import os.path

class IdentifiedUnit(object):
    '''
    Container for a single identified unit, spikes and name.
    
    Attributes
    ----------
    parent : DataCollection
      collection of raw data to which this unit's spikes belong
    unit_name : string
      unique name for this unit
      based on first filename and channel/id
    lag : float
      lag time between neural and movement data
      positive lag implies neural event precedes kinematic
      
    Parameters
    ----------
    parent : DataCollection
      see Attributes
    unit_names : sequence of string
      names of the units in each file of `parent`
    lag : float
      see Attributes    
    '''
    def __init__(self, unit_names, lag, parent):
        self.parent = parent
        self.unit_names = unit_names
        self.lag = lag
        self.spikes = []
        self.get_spikes()

        assert len(self.spikes) == len(self.parent.HoldAStart)

    def get_full_name(self):
        '''
        Return the full name of the unit derived from unit name and lag.
        
        Returns
        -------
        full_name : string
          full name of unit including lag, in format "Unit001_1_100ms"
        '''
        first_file = os.path.split(self.parent.files[0])[-1]
        first_file = os.path.splitext(first_file)[0].replace('.', '_')
        first_unit = self.unit_names[0]
        return first_file + '_' + first_unit + '_%dms' % int(self.lag * 1e3)

    def get_spikes(self):
        '''
        Populates spike data from .mat file
        '''
        # pick correct file handling routine
        if self.parent.version == 'VR_RTMA_1.0':
            opener = CenOut_VR_RTMA_10_File
        elif self.parent.version == '3d_data_2.1':
            opener = CenOut_3d_data_21_File
        else:
            raise ValueError("Not a recognized format.")

        for file_name, unit_name in zip(self.parent.files, self.unit_names):
            print "Fetching %s in %s" % (file_name, unit_name)
            # open file and grab sorted spikes
            exp_file = opener(file_name)
            file_spikes = exp_file.sort_spikes(unit_name, lag=self.lag)
            self.spikes.extend(file_spikes)
            print "Now have %d trials" % (len(self.spikes))
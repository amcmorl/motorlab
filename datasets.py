import os
import numpy as np
from socket import gethostname
from ConfigParser import ConfigParser
from git import Repo
from warnings import warn
from glob import glob
from motorlab.vr.data_files import list_units

# datasets
# osmd = offline sorted multiday
hostname = gethostname()
home = os.environ['HOME']
pitt_dir = home + '/files/pitt'
raw_dir = pitt_dir + '/raw_data'
datasets_config = raw_dir + '/datasets.conf'

# has to be able to cope with branch specific datasets git-using repositories
# and non-git-using repositories

# on one host there is a given root/branch combination

class DatasetSimpleUnitID(object):
    def __init__(self, root, branch, days):
        self.root = raw_dir + root
        self.branch = branch
        self.days = [x.strip() for x in days.split(',')]
        self.monkey = monkey
        
    def check_git(self):
        if self.branch != 'no git':
            repo = Repo(self.root)
            if self.branch == 'any':
                warn("Active branch in data repository is %s" % \
                         (repo.active_branch))
            elif not str(repo.active_branch) == self.branch:
                raise EnvironmentError("Active branch in data repository is"
                                       "%s, but %s is required." %
                                       (repo.active_branch, self.branch))
     
    def get_files(self):
        self.check_git()
        file_pat = self.monkey.capitalize() + '.VR.*.CenterOut.mat'
        files = []
        for day in self.days:
            files.extend(glob(self.root + '/' + day + '/' + file_pat))
        files.sort()
        return files
                    
    def get_units(self):
        file0 = self.get_files()[0]
        return list_units(file0)
    
# construct datasets database from config file
config = ConfigParser()
config.read(datasets_config)
datasets = {}
for dataset_name in config.sections():
    if config.get(dataset_name, 'type') == 'simple':
        root = config.get(dataset_name, 'root')
        days = config.get(dataset_name, 'days')
        monkey = config.get(dataset_name, 'monkey')
        git_branch = config.get(dataset_name, 'git_branch')
        datasets[dataset_name] = DatasetSimpleUnitID(root, git_branch, days)
    else:
        
        warn('Cannot handle non-simple dataset types yet.')
    
def pick_units_by_number(units, cond='>100'):
    '''
    Select a subset of units based on `condition` relative to their unit_num.
    '''
    out = []
    for unit in units:
        unit_num, chan_num = unit.split('_')
        unit_num = int(unit_num[4:]) # assumes has 'Unit' on the front
        if eval(str(unit_num) + cond):
            out.append(unit)
    return out

def get_m1_units_from_unit_name(unit_name):
    '''selects units 100 - 199, which are from the M1 array, at least in Frank
    '''
    return np.asarray([x[4] == '1' for x in unit_name])
    
def get_nice_name(unit_name):
    '''
    Convert a name in the Unit012_3 format to 12c format.
    '''
    channel, sub = unit_name.split('_')
    if channel.startswith('Unit'):
        channel = str(int(channel[4:]))
    offset = ord('a') - ord('1')
    sub = chr(ord(sub) + offset)   
    return channel + sub

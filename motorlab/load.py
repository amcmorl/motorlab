from git import Repo
from warnings import warn
from glob import glob
from socket import gethostname
from motorlab.data_files import list_units

'''
'''
def check_git_status(branch='any'):
    '''
    Check git branch of `location` on hostname.
    '''
    hostname = gethostname()
    if hostname in uses_git:
        repo = Repo(locations[hostname])
        if branch == 'any':
            warn("Active branch in data repo is %s" % (repo.active_branch))
        elif not repo.active_branch == branch:
            raise EnvironmentError("Active branch in data repo is"
                   "%s; %s required." % (repo.active_branch, branch))

def get_file_list(dataset='offline sorted multiday'):
    '''
    '''
    dset_name = aliases[dataset]
    dset = datasets[dset_name]
    check_git_status(branch=dset['git'])
    data_dir = locations[gethostname()]
    file_pat = 'Frank.VR.*.CenterOut.mat'
    files = []
    if 'files' in dset.keys():
        files.extend(dset['files'])
    if 'days' in dset.keys():
        for day in dset['days']:
            files.extend(glob(data_dir + day + '/' + file_pat))
    files.sort()
    return files

def get_units_list(dataset='offline sorted multiday'):
    file0 = get_file_list(dataset=dataset)[0]
    return list_units(file0)


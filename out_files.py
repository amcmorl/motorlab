import os
import time
import motorlab.tuning
#import motorlab.network

project_dirs = {#'network' : motorlab.network.proj_dir,
                'tuning' : motorlab.tuning.proj_dir}

def make_today_dir(project='tuning'):
    outputs = project_dirs[project] + '/run'
    today = outputs + '/' + time.strftime('%y%m%d')
    if not os.path.exists(today):
        os.mkdir(today)
    return today

def now_str():
    return time.strftime('%H%M%S')

def get_out_name(save_dir, what, ext=""):
    oname = save_dir + '/' + what + '_' + now_str()
    if ext != "":
        oname += ('.' + ext.lstrip('.')) # make sure there's only one '.'
    return oname

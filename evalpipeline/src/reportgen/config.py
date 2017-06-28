'''
Created on May 7, 2017

@author: arvind
'''

class Config(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        template_vars = {}
        template_vars['team_name'] = 'DSRUF'
        template_vars['submission_id'] = 'AadX149d'
        template_vars['submission_cnt'] = 2
        template_vars['submission_total'] = 4
        self.template_vars = template_vars
        
        self.template_dir = 'template/'
        self.datadir = '../../data/NIST_data_20170120/'
        self.indir = '../../income/dummy/'
        self.outdir = '../../outputs/dummy/'
        self.outdir = 'report/'

'''
Created on May 2, 2017

@author: arvind
'''

class EvaluationException(object, Exception):
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        self.message = params
    
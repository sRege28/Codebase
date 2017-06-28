'''
Created on May 1, 2017

@author: arvind
'''

import shapefile
import numpy as np
from numpy import dtype
from __builtin__ import str

class AlignModel(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def align(self):
        pwd = '/home/arvind/MyStuff/Coursework/PDS/NIST-DATA/NIST_data_20170120/'
        sf_trees_recs = []
        sf_trees = shapefile.Reader(pwd + 'vector/Final Tagged Trees.shp')
        no_of_trees = len(sf_trees.records())
        sf_trees_recs = sf_trees.records()
        sf_trees = shapefile.Reader(pwd + 'vector/Final Untagged Trees.shp')
        no_of_trees = no_of_trees + len(sf_trees.records())
        sf_trees_recs =  sf_trees_recs + sf_trees.records()
        sf_stems = shapefile.Reader(pwd + 'vector/Final Tree Center Points.shp')
        no_of_stems = len(sf_stems.records())
        
        alignment_matrix = np.empty((no_of_trees+1, no_of_stems+1), dtype=object)
        labels = ['DC']
        for rec in sf_stems.records():
            labels.append(rec[0])
        alignment_matrix[0] = labels 
        for i in range(no_of_trees):
            logits = np.random.dirichlet(np.ones(no_of_stems),size=1).astype(object).tolist()[0]
            try:
                logits.insert(0, str(sf_trees_recs[i][0]))
            except:
                print np.array(logits).shape
                print logits, i
                raise
            alignment_matrix[i+1] = logits#[sf_trees.record(i)[0]] + np.random.dirichlet(np.ones(no_of_stems),size=1).tolist()
        np.savetxt('/home/arvind/shpout/alignment_out.csv', alignment_matrix, fmt='%s', delimiter=',')
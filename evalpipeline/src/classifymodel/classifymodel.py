'''
Created on May 1, 2017

@author: arvind
'''
import shapefile
import numpy as np

class ClassifyModel(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    
    def classify(self):
        pwd = '/home/arvind/MyStuff/Coursework/PDS/NIST-DATA/NIST_data_20170120/'
        species = {}
        sf_trees_recs = []
        sf_trees = shapefile.Reader(pwd + 'vector/Final Tagged Trees.shp')
        for rec in sf_trees.records():
            species[rec[2]] = 'D'
        no_of_trees = len(sf_trees.records())
        sf_trees_recs = sf_trees.records()
        sf_trees = shapefile.Reader(pwd + 'vector/Final Untagged Trees.shp')
        no_of_trees = no_of_trees + len(sf_trees.records())
        for rec in sf_trees.records():
            species[rec[2]] = 'D'        
        sf_trees_recs = sf_trees_recs +  sf_trees.records()

        no_of_species = len(species.keys())
        labels = ['DC'] + species.keys()
        classification_matrix = np.empty((no_of_trees+1, no_of_species+1), dtype=object)
        print classification_matrix.shape
        classification_matrix[0] = labels
        for i in range(no_of_trees):
            logits = np.random.dirichlet(np.ones(no_of_species),size=1).astype(object).tolist()[0]
            try:
                logits.insert(0, str(sf_trees_recs[i][0]))
            except:
                print np.array(logits).shape
                print logits, i
                raise
            classification_matrix[i+1] = logits#[str(sf_trees_recs[i][0])] + np.random.dirichlet(np.ones(no_of_species),size=1).astype(object).tolist()
        np.savetxt('/home/arvind/shpout/classification_out.csv', classification_matrix, fmt='%s', delimiter=',')
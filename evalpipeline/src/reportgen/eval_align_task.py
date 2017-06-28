'''
Created on May 2, 2017

@author: arvind
'''
import numpy as np
import shapefile
import collections

class AlignMetric(object):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''
        self.config = config
        self.score = 0.0
        self.plotwise_accuracy = {}

    def evaluate(self):
        pwd = self.config.datadir
        sf_trees = shapefile.Reader(pwd + 'vector/Final Tagged Trees.shp')
        no_of_trees = len(sf_trees.records())
        sf_trees_recs = sf_trees.records()
        sf_trees = shapefile.Reader(pwd + 'vector/Final Untagged Trees.shp')
        no_of_trees = no_of_trees + len(sf_trees.records())
        sf_trees_recs =  sf_trees_recs + sf_trees.records()
        plotmap = {}
        # mapping between tree id and plot number
        for rec in sf_trees_recs:
            plotmap[str(rec[0]).replace(" ", "")] = rec[-1]
        
        alignment_matrix = np.loadtxt(self.config.indir + 'alignment_out.csv', dtype=object, delimiter=',')
        alignment_map = {}
        # reading the labels given by the participant to get order
        labels_map = dict(enumerate(alignment_matrix[0,1:]))
        for i in np.arange(1, alignment_matrix.shape[1]):
            row = alignment_matrix[i,:]
            key = row[0]
            row = row[1:]
            alignment_map[key.replace(" ","")] = row
        
        # calculating the trace of the matrix isnt straight forward since order isnt constrained
        # hence using a map data structure to get the right alignment for each stem id -> tree id mapping
        sum_correct_probabilities = 0
        for idx in labels_map.keys():
            key = labels_map[idx].replace(" ","")
            if key in alignment_map:
                sum_correct_probabilities += float(alignment_map[key][idx])
                if plotmap[key] not in self.plotwise_accuracy:
                    self.plotwise_accuracy[plotmap[key]] = 0.0
                self.plotwise_accuracy[plotmap[key]] += float(alignment_map[key][idx])    
            else:
                print key
        no_of_trees = alignment_matrix.shape[0]
        self.score = sum_correct_probabilities/float(no_of_trees)
        self.plotwise_accuracy = collections.OrderedDict(sorted(self.plotwise_accuracy.items(), key=lambda x: float(x[0].replace('L', ''))))
        return self.score
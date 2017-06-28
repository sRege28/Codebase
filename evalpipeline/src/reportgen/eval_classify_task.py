'''
Created on May 3, 2017

@author: arvind
'''

import shapefile
import numpy as np
from sets import Set

class ClassifyMetric(object):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''
        self.config = config
        self.specieswise_accuracy = {}
        self.score = 0.0
        self.rank_1_acc = 0.0
        self.true_positives = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.species_list =[]
        
    #TODO check for singularity [does'nt occur usually!]
    def get_precision_map(self):
        return dict((k, float(self.get_tp(k)) / (self.get_tp(k) + self.get_fp(k, default=0.0001))) for k in self.species_list)
    
    def get_recall_map(self):
        return dict((k, float(self.get_tp(k)) / (self.get_tp(k) + self.get_fn(k, default=0.0001))) for k in self.species_list)
    
    
    # wrapper fns that help us avoid checking everywhere
    def get_fp(self, key, default=0.0):
        if key in self.false_positives:
            return self.false_positives[key]
        else:
            return default

    def get_tp(self, key, default=0.0):
        if key in self.true_positives:
            return self.true_positives[key]
        else:
            return default
        
    def get_fn(self, key, default=0.0):
        if key in self.false_negatives:
            return self.false_negatives[key]
        else:
            return default
        
    def evaluate(self):
        pwd = self.config.datadir
        sf_trees = shapefile.Reader(pwd + 'vector/Final Tagged Trees.shp')
        no_of_trees = len(sf_trees.records())
        sf_trees_recs = sf_trees.records()
        sf_trees = shapefile.Reader(pwd + 'vector/Final Untagged Trees.shp')
        no_of_trees = no_of_trees + len(sf_trees.records())
        sf_trees_recs =  sf_trees_recs + sf_trees.records()
        speciesmap = {}
        #getting tree id to species mapping; also removing spaces in treeid since there are discrepancies in the dataset 
        for rec in sf_trees_recs:
            speciesmap[str(rec[0]).replace(" ", "")] = rec[2]
        classification_matrix = np.loadtxt(self.config.indir +'classification_out.csv', dtype=object, delimiter=',')
        classification_map = {}
        confusion_matrix = np.zeros((classification_matrix.shape[1]-1, classification_matrix.shape[1]-1))
        
        #reading the species label submitted by the participants
        # and making a map based on index ; also inverting the index for 2 way lookup
        labels_map = dict(enumerate(classification_matrix[0,1:]))
        labels_map_inv = {v: k for k, v in labels_map.iteritems()}
        for i in np.arange(1, classification_matrix.shape[0]):
            row = classification_matrix[i,:]
            key = row[0]
            row = row[1:]
            classification_map[key.replace(" ","")] = row
        
        #calculating the log loss using the correct species's logit
        sum_correct_probabilities = 0
        for idx in classification_map.keys():
            key = speciesmap[idx]
            if key in labels_map_inv:
                species_label = speciesmap[idx]
                confusion_matrix[np.argmax(classification_map[idx]), labels_map_inv[key]] += 1.0;
                if np.argmax(classification_map[idx]) == labels_map_inv[key]:
                    self.rank_1_acc += 1.0
                    if species_label not in self.true_positives:
                        self.true_positives[species_label] = 0.0
                    self.true_positives[species_label] += 1.0
                else:
                    if species_label not in self.false_negatives:
                        self.false_negatives[species_label] = 0.0
                    self.false_negatives[species_label] += 1.0
                    if labels_map[np.argmax(classification_map[idx])] not in self.false_positives:
                        self.false_positives[labels_map[np.argmax(classification_map[idx])]] = 0.0
                    self.false_positives[labels_map[np.argmax(classification_map[idx])]] += 1.0
                sum_correct_probabilities += np.log(float(classification_map[idx][labels_map_inv[key]]))
                if key not in self.specieswise_accuracy:
                    self.specieswise_accuracy[key] = 0.0
                self.specieswise_accuracy[key] += np.log(float(classification_map[idx][labels_map_inv[key]]))   
            else:
                #present in Tagged Trees but absent in TreeCenterPoints file : NOOP
                print key
        self.species_list = list(Set(speciesmap.values()))
        no_of_trees = classification_matrix.shape[0]
        self.score = sum_correct_probabilities/float(no_of_trees)
        self.rank_1_acc = self.rank_1_acc / float(no_of_trees)
        self.confusion_matrix = confusion_matrix
        self.species_list = labels_map_inv.keys()
        return self.score
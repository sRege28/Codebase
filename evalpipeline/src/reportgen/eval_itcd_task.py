'''
Created on May 1, 2017

@author: arvind
'''
import cv2
import numpy as np
from munkres import Munkres, print_matrix
import matplotlib.pyplot as plt
from collections import OrderedDict
import shapefile
from shapely.geometry import Polygon
#placeholder for nishant to check in his code

class MyPolygon:
    def __init__(self, points):
        self.intersectionArea = 0
        self.polygon = Polygon(points).buffer(0) 
        
    def getPolygonForm(self):
        return self.polygon
        
    def setIntersectionArea(self, anotherPolygon):        
        self.intersectionArea = self.intersectionArea + self._intersection(anotherPolygon).area
        
    def area(self):
        return self.getPolygonForm().area
    
    def union(self, anotherPolygon):
        return self.getPolygonForm().union(anotherPolygon.getPolygonForm())
    
    def intersection(self, anotherPolygon):
        return self._intersection(anotherPolygon).area - self.intersectionArea

    def _intersection(self, anotherPolygon):
        return self.getPolygonForm().intersection(anotherPolygon.getPolygonForm())
    
    def intersects(self, anotherPolygon):
        return self.getPolygonForm().intersects(anotherPolygon.getPolygonForm())

class DelineationMetric(object):
    '''
    classdocs
    '''


    subtractConstant = 999999
    
    def __init__(self):
        self.numberOfPlots = 0
        self.itcPolygonList = []
        self.predictedPolygonList = []
        self.jaccardSimilarityForAllPlots = 0       
        self.RecallForAllPlotsList = []
        self.topPolygons = {}
        self.falsePositives = 0
        self.falseNegatives = 0
        self.truePositives = 0
        self.plotLevelTruePositives = []
        self.plotLevelFalsePositives = []
        self.plotLevelFalseNegatives = []
        self.assigmentMap = []

    
    def performAssignment(self, costMatrix):
        if len(costMatrix) == 0:
            indexes = []
        else:
            m = Munkres()
            indexes = m.compute(costMatrix)
        #print_matrix(costMatrix, msg='Maximum cost through this matrix:')
        assignmentList = []
        for row, column in indexes:                                 
            assignmentList.append((row, column))  
        return assignmentList
    
    def cleanUpData(self, groundTruthPoly, PredictedPolys):
        CleanedGroundTruth = []
        for index , poly in enumerate(groundTruthPoly):
            if cv2.contourArea(poly) > 100 and cv2.contourArea(poly) < 8000:
                polyPoints = [tuple(l) for l in poly[0].tolist()]
                if len(polyPoints) >= 3:
                    CleanedGroundTruth.append(poly)  
        CleanedPredictedPolys = []
        for index , poly in enumerate(PredictedPolys):
            if cv2.contourArea(poly) > 100 and cv2.contourArea(poly) < 8000:
                polyPoints2 = [tuple(l) for l in poly[0].tolist()]
            #print polyPoints2
                if len(polyPoints2) >= 3:
                    CleanedPredictedPolys.append(poly) 
        return (CleanedGroundTruth, CleanedPredictedPolys)
     
    def checkIfPolygonsIntersect(self):
        count = 0        
        for index in range(len(self.predictedPolygonList)):
            for index2 in range(index + 1, len(self.predictedPolygonList)):
                if self.predictedPolygonList[index].intersects(self.predictedPolygonList[index2]):
                    count = count + 1
        return count
     
     
    def formingPolygonsFromPoints(self, itcPolys, predictedPolygons):
        for itc in itcPolys:
            polygonPoints = [tuple(l) for l in itc[0].tolist()]           
            groundTruthPolygon = MyPolygon(polygonPoints)
            self.itcPolygonList.append(groundTruthPolygon)
            
        for pred in predictedPolygons:
            predPoints = [tuple(l) for l in pred[0].tolist()]               
            predictedPoly = MyPolygon(predPoints)
            self.predictedPolygonList.append(predictedPoly)
            self.checkIfPolygonsIntersect()
        
    ## main method which initiates the the computation of hungarian assignment and jaccard coefficient
    def calculateHungarianAssignment(self,plotNo,itcPolys, predictedPolygons):
         # cleans up the data so that no polygons are there having less than 3 points
         itcPolys, predictedPolygons = self.cleanUpData(itcPolys, predictedPolygons)
         
         self.predictedPolygonList = []
         self.itcPolygonList = []
         
         ### converting the polygons returned dron findContours to be used by shapely
         self.formingPolygonsFromPoints(itcPolys, predictedPolygons)
         
         ### check if a the submitted Predicted polygons overlap with each other
         print 'Error as the predictedPolygon Intersects: ' + str(self.checkIfPolygonsIntersect())
         
         areaCostMatrix = []
         for groundTruthPolygon in self.itcPolygonList:                                            
             intersectionAreas = []
             for predictedPoly in self.predictedPolygonList:                            
                 intersectionArea = groundTruthPolygon.intersection(predictedPoly)
                 intersectionAreas.append(DelineationMetric.subtractConstant - intersectionArea)
             areaCostMatrix.append(intersectionAreas)
         try:         
             assignedPolyList = self.performAssignment(areaCostMatrix)
         except:
             print areaCostMatrix
             raise
         similarityScoreForOnePlot = self.calculateJaccardCoefficients(plotNo, predictedPolygons, assignedPolyList, areaCostMatrix)
         self.jaccardSimilarityForAllPlots =  self.jaccardSimilarityForAllPlots + similarityScoreForOnePlot
         self.numberOfPlots = self.numberOfPlots + 1
         self.setHistogramOfRecall(assignedPolyList)
         self.assigmentMap = assignedPolyList
         return assignedPolyList, itcPolys, predictedPolygons 
     
    def setHistogramOfRecall(self, assignedPolys):        
         for matchingPair in assignedPolys:
             self.RecallForAllPlotsList.append(self.predictedPolygonList[matchingPair[1]].area() / self.itcPolygonList[matchingPair[0]].area())                    
         
    ### Returns the final overall jaccard score
    def getFinalJaccardScore(self):
        try:
            return self.jaccardSimilarityForAllPlots / self.numberOfPlots
        except ZeroDivisionError:
            print 'Please Run the Hungarian Assignement First' 
            
    # Get the Histogram for Recall where Y axsi denotes the count and x axsi specifies the jaccoard score
    def getHistogramForRecall(self):
        bins = values = np.linspace(0, 1, 11, endpoint = True)
        arrayList  = np.asarray(self.RecallForAllPlotsList)
        #print len(self.RecallForAllPlotsList)
        #print len(arrayList)
        plt.hist(arrayList, bins)       
        plt.xlabel('Jaccard Scores')
        plt.ylabel('Count of Predicted Polygons')
        plt.xticks(bins)
        plt.title('Histogram of Recall')
        return plt       
                                     
         
    def calculateJaccardCoefficients(self, plotNo, predictedpolygons, assignedPolyList, areaMatrix):
        similarityCoefficient = 0
        tp, fp, fn = self.truePositives, self.falsePositives, self.falseNegatives
        for matchingPair in assignedPolyList:
            actualPoly = self.itcPolygonList[matchingPair[0]]
            predPoly = self.predictedPolygonList[matchingPair[1]]
            unionArea  = actualPoly.union(predPoly).area
            intersectionArea = DelineationMetric.subtractConstant - areaMatrix[matchingPair[0]][matchingPair[1]]
            if intersectionArea!=0.0:
                self.truePositives = self.truePositives + intersectionArea
                self.falsePositives = self.falsePositives + predPoly.area() - intersectionArea
            self.falseNegatives = self.falseNegatives + actualPoly.area() - intersectionArea
            similarScore = intersectionArea / unionArea
            p = (plotNo, matchingPair[1])
            self.topPolygons[p] = (similarScore, predictedpolygons[matchingPair[1]])
            similarityCoefficient += similarScore
        self.plotLevelTruePositives.append(self.truePositives - tp)             
        self.plotLevelFalsePositives.append(self.falsePositives - fp)
        self.plotLevelFalseNegatives.append(self.falseNegatives - fn)
        if len(assignedPolyList) == 0 :
            return 0.0 #avoid singularity
        return similarityCoefficient / (len(assignedPolyList))
     
    
    #### function to get the top 4 worst 4 predicted polygons 
    def getTopPolygons(self):
        sorted_dict_by_score = OrderedDict(sorted(self.topPolygons.items(), key=lambda x: x[1][0])) 
           
        worstPredictions = sorted_dict_by_score.items()[0 : 5]
        bestPredictions = sorted_dict_by_score.items()[-5: ]
        return (worstPredictions, bestPredictions)
    
    ### returns the consudion matrix 
    def getConfusionMatrix(self):
        return (self.truePositives, self.falsePositives, self.falseNegatives)

    ### returns the consudion matrix on plot level
    def getConfusionMatrixOnPlotLevel(self):       
        return (self.plotLevelTruePositives, self.plotLevelFalsePositives, self.plotLevelFalseNegatives)
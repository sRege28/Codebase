'''
Created on May 2, 2017

@author: arvind
'''
from itcdmodel import basemodel
import cv2
import numpy as np
from matplotlib import pyplot as plt


class WSModel(basemodel.BaseModel):
    '''
    classdocs
    '''


    def __init__(self, plotno):
        '''
        Constructor
        '''
        super(WSModel, self).__init__(plotno)
   
    def execute(self):
        img = self.img
        #img = cv2.resize(img, (80, 80))
        
        delta= 70
        lower_gray = np.array([delta, delta,delta])
        upper_gray = np.array([255-delta,255-delta,255-delta])
        
        mask = cv2.inRange(img, lower_gray, upper_gray)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        
        
        HSV_img = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        hue = HSV_img[:, :, 0]
        
        
        hist = cv2.calcHist([hue],[0],None,[256],[0,256])
        hist= hist[1:, :] #suppress black value
        elem = np.argmax(hist)
        
        
        tolerance=10
        lower_gray = np.array([elem-tolerance, 0,0])
        upper_gray = np.array([elem+tolerance,255,255])
        # Threshold the image to get only selected
        mask = cv2.inRange(HSV_img, lower_gray, upper_gray)
        # Bitwise-AND mask and original image
        res2 = cv2.bitwise_and(img,img, mask= mask)
        
        titles = ['Original Image',' Final Result']
        images = [img,res2]
        for i in xrange(2):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
        imgray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(thresh,kernel,iterations=1)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.min(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        markers = cv2.watershed(img, markers)
            #shrink = cv2.resize(out,(width, height), interpolation = cv2.INTER_CUBIC)
                #print shrink
        img[markers == -1] = [255,0,0]
        
        
        
        mask = []
        polys = []
        img2 = np.zeros_like(img)
        for i in range(ret):
            mask = (markers == i)
            mask = mask.T
            Points = []
            for y in range(markers.shape[0]):
                for x in range(markers.shape[1]):
                    if mask[y][x] == True:
                        Points.append([y, x])
            approx_curve = cv2.approxPolyDP(np.array(Points), 1, False)   
            conv_hull = cv2.convexHull(approx_curve)   
            polys.append(conv_hull)
       
        return polys
        
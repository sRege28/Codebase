'''
Created on May 1, 2017

@author: arvind
'''
from itcdmodel import basemodel

import numpy as np
import cv2
import georasters as gr
from matplotlib import pyplot as plt
import gdal
from spectral import principal_components as pc
import shapefile
from osgeo import osr
from shapely.geometry import Polygon
from scipy.ndimage import gaussian_filter

class LoGModel(basemodel.BaseModel):
    '''
    classdocs
    '''


    def __init__(self, plotno):
        '''
        Constructor
        '''
        super(LoGModel, self).__init__(plotno)
    
    def execute(self):
        log = self.auto_laplacian(np.invert(self.img_fc.astype('uint8')))
        pad = 3
        canny = cv2.copyMakeBorder(log, pad, pad, pad, pad,cv2.BORDER_CONSTANT, value=1)
        contours = cv2.findContours(canny.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.imgcp(), contours[1], -1, color=[0,255,255], thickness=2)
        non_local_filter = self.max_convolve(self.img_fc, ksize=3)
        geodesic_distance_overall = np.zeros(self.img_dims)
        mask = np.zeros(self.img_dims)
        for cnt in contours[1]: 
            if len(cnt)<3 or cv2.contourArea(cnt) < 50 or cv2.contourArea(cnt) > 10000:
                continue
            y, x, h, w =  cv2.boundingRect(cnt)
            cnt = np.append(cnt, [cnt[0]], axis = 0)
            cv2.fillPoly(mask, pts=[cnt], color=100)
            cv2.fillConvexPoly(mask, points=cnt, color=100)
            geodesic_dist = self.geodesic_distance(mask, ksize=3, sx = x, sy=y, ex = x+w, ey=y+h)
            geodesic_distance_overall+= geodesic_dist
            mask[mask!=0] = 0
        spatial_maximum = np.zeros(self.img_dims)
        mask = np.zeros(self.img_dims)
        prob_cnt = mask
        for cnt in contours[1]: 
            if len(cnt)<3 or cv2.contourArea(cnt) < 50 or cv2.contourArea(cnt) > 10000:
                continue
            y, x, h, w =  cv2.boundingRect(cnt)
            cnt = np.append(cnt, [cnt[0]], axis = 0)
            #cv2.fillPoly(mask, pts=[cnt], color=100)
            cv2.fillConvexPoly(mask, points=cnt, color=100)
            regional_max = self.regional_maxima(mask, ksize=3, sx = x, sy=y, ex = x+w, ey=y+h)
            if(np.sum(regional_max)==0):
                prob_cnt = (mask, x, y, w, h, cnt)
                print x, y, x+w, y+h
                break
            spatial_maximum+= regional_max
            mask[mask!=0] = 0
        overall_marker_filters = self.filter_concat(spatial_maximum, non_local_filter)
        filtered_contours = []
        mask = np.zeros(self.img_dims)
        overall_markers = np.zeros(self.img_dims)
        color_count = 1
        
        for cnt in contours[1]:
            if len(cnt)<3:
                continue
            if cv2.contourArea(cnt) > 10000:
                continue
            if cv2.contourArea(cnt) < 20:
                continue
            cnt = np.append(cnt, [cnt[0]], axis = 0)
            cv2.fillPoly(mask, pts=[cnt], color=1)
            cv2.fillConvexPoly(mask, points=cnt, color=1)
            if np.sum(np.logical_and(mask, non_local_filter))>0:
                #cnt = cv2.convexHull(cnt, 0, True)
                filtered_contours.append(cnt)
                cv2.fillPoly(overall_markers, pts=[cnt], color=color_count)
                cv2.fillConvexPoly(overall_markers, points=cnt, color=color_count)
                color_count = color_count + 1
        print len(filtered_contours), len(contours[1])
        overallmarkers = np.zeros(self.img_dims)
        i = 1
        granular_contours = []
        for cnt in filtered_contours:
            if len(cnt)<3 or cv2.contourArea(cnt)<1000:
                granular_contours.append(cnt)
                continue
            msk = np.zeros(self.img_dims).astype('uint8')
            cv2.drawContours(msk,[cnt], -1 , color=1, thickness=-2)
            lowpass = gaussian_filter(self.img_fc.astype('float64'), 10)
            gauss_highpass = self.img_fc.astype('float64') - lowpass
            gauss_highpass[msk!=1] = 0
            gauss_highpass = cv2.erode(gauss_highpass,np.ones((2,2)))
            cont = cv2.findContours(gauss_highpass.astype('int32'), mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_SIMPLE, offset=(10,-10))
            cont = [cnt for cnt in cont[1] if cv2.contourArea(cnt)<3000]
            granular_contours = granular_contours + cont

        granular_contours = [cv2.convexHull(cnt) for cnt in granular_contours if cv2.contourArea(cnt)<3000 and len(cnt)>3]
        
        repeat_remover = np.zeros(self.img_dims)
        removed_contours = []
        filtered_granular_contours = []
        for i in range(len(granular_contours)):
            cnt1 = granular_contours[i]
            totalIntersectArea = 0
            x,y,w,h = cv2.boundingRect(cnt1)
            brect1 = Polygon(np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h), (x,y)]))
            for j in range(len(granular_contours)):
                if j==i:
                    continue
                cnt2 = granular_contours[j]
                x,y,w,h = cv2.boundingRect(cnt2)
                brect2 = Polygon(np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h), (x,y)]))
                if brect1.contains(brect2):
                    intersectArea = brect1.intersection(brect2).area
                    totalIntersectArea = totalIntersectArea + intersectArea
            if totalIntersectArea < 0.70 * brect1.area:
                filtered_granular_contours.append(cnt1)
            else:
                removed_contours.append(cnt1)
            
            if totalIntersectArea >= 0.99 * brect1.area:
                mask = np.zeros(self.img_dims)
                cv2.fillPoly(mask, pts=[cnt1], color=1)
                cv2.fillConvexPoly(mask, points=cnt1, color=1)
                if np.sum(repeat_remover[mask==1]) > 0.88 * np.sum(mask):
                    filtered_granular_contours.append(cnt1)
                cv2.fillPoly(repeat_remover, pts=[cv2.approxPolyDP(cnt1, 0.5, True)], color=1)
                cv2.fillConvexPoly(repeat_remover, points=cv2.approxPolyDP(cnt1, 0.5, True), color=1)

        return filtered_granular_contours
            
            
            
    def auto_laplacian(self, gray):
        gray_gauss = cv2.GaussianBlur(gray.astype('float64'),(0,0), 10)
        gray = gray - gray_gauss
        if len(gray.shape)>2:
            gray = cv2.cvtColor(gray.astype('float32'), cv2.COLOR_BGR2GRAY)
        else:
            gray = gray.astype(float)      
        kernel_size = 3
        scale = 15
        delta = 1
        ddepth = cv2.CV_64F
        gray_lap = cv2.Laplacian(gray.astype('float'),ddepth,ksize = kernel_size,scale = scale,delta=delta)
        _,dst = cv2.threshold(np.abs(gray_lap).astype('uint8'),0,1, cv2.THRESH_OTSU) 
        return dst
    
    def max_convolve(self, ip, ksize=3, sx=0, sy=0, ex=-1, ey = -1):
        #general template for 2d convolution with bound checking
        (iW, iH) = (ip.shape[0], ip.shape[1])
        output = np.zeros((iW, iH))
        (kW, kH) = (ksize, ksize)
        #pad by how much the kernel needs to also cover the edges
        pad = (kW-1) / 2
        #if start is mentioned add the padding
        if sx!=-1 and sy !=-1:
            sx += pad
            sy += pad
        #if end is not mentioned default to width and height
        #else add padding to the end 
        if ex==-1 or ey ==-1:
            ex,ey = iW, iH
        else:
            ex += pad 
            ey += pad
        
        #perform the padding
        ip = cv2.copyMakeBorder(ip, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        for x in np.arange(sx, ex):
            for y in np.arange(sy, ey):
                roi = ip[x:x+kW,y:y+kH]
                #specific logic for max convolve
                max_idx = np.unravel_index(np.argmax(roi),roi.shape)
                if (max_idx[0] == kW/2 and max_idx[1] == kH/2) or roi[kW/2,kH/2] == roi[max_idx[0],max_idx[1]]:
                    _x = x - pad + kW/2
                    _y = y - pad + kH/2
                    output[_x,_y] = 1
        return output
    
    def filter_concat(self, ip, spatial_max, ksize=3):
        #general template for 2d convolution with bound checking
        (iW, iH) = (ip.shape[0], ip.shape[1])
        output = np.zeros((iW,iH))
        (kW, kH) = (ksize, ksize)
        #pad by how much the kernel needs to also cover the edges
        pad = (kW-1) / 2
        
        #perform the padding
        ip = cv2.copyMakeBorder(ip, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        #to check if recursive pass needed
        for x in np.arange(0, iW):
            for y in np.arange(0, iH):
                if x+kW>iW or y+kH>iH:
                    #dirty fix to avoid cases where we overrun the array
                    continue
                roi = ip[x:x+kW,y:y+kH]
                #specific logic for filter concat
                try:
                    if roi[roi.shape[0]/2, roi.shape[1]/2]>0:
                        #now check if spatial tree top can be found at a distance of 3 units
                        spatial_roi = spatial_max[x:x+kW, y:y+kH]
                        if np.sum(spatial_roi)>0:
                            _x = x - pad + kW/2
                            _y = y - pad + kH/2
                            output[_x, _y] = 1
                except:
                    print roi, x, x+kW, y, y+kH
                    return roi
        return output
    
    
    
    def geodesic_distance(self, ip, ksize=3, sx=0, sy=0, ex=-1, ey = -1, recurseCnt=0):
        #general template for 2d convolution with bound checking
        (iW, iH) = (ip.shape[0], ip.shape[1])
        output = np.copy(ip)
        (kW, kH) = (ksize, ksize)
        #pad by how much the kernel needs to also cover the edges
        pad = (kW-1) / 2
        #if start is mentioned add the padding
        if sx!=-1 and sy !=-1:
            sx += pad
            sy += pad
        #if end is not mentioned default to width and height
        #else add padding to the end 
        if ex==-1 or ey ==-1:
            ex,ey = iW, iH
        else:
            ex += pad 
            ey += pad
        
        #perform the padding
        ip = cv2.copyMakeBorder(ip, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        #to check if recursive pass needed
        exchangeHappened = False
        for x in np.arange(sx, ex):
            for y in np.arange(sy, ey):
                if x+kW>iW or y+kH>iH:
                    #dirty fix to avoid cases where we overrun the array
                    continue
                roi = ip[x:x+kW,y:y+kH]
                #specific logic for geodesic distance
                try:
                    if roi[roi.shape[0]/2, roi.shape[1]/2]>0:
                        _x = x - pad + kW/2
                        _y = y - pad + kH/2
                        minval = np.min(roi)
                        if output[_x,_y] > minval + 1:
                            exchangeHappened = True
                        output[_x, _y] = minval + 1
                except:
                    print roi, x, x+kW, y, y+kH
                    return roi
        if exchangeHappened:
            if recurseCnt<30:
                return self.geodesic_distance(output, sx=sx, sy=sy, ex = ex, ey = ey, recurseCnt=recurseCnt+1)
            else:
                print 'potentially unconverged solution!!!'
        return output
    
    def regional_maxima(self, ip, ksize=3, sx=0, sy=0, ex=-1, ey = -1):
        #general template for 2d convolution with bound checking
        (iW, iH) = (ip.shape[0], ip.shape[1])
        output = np.zeros((iW,iH))
        (kW, kH) = (ksize, ksize)
        #pad by how much the kernel needs to also cover the edges
        pad = (kW-1) / 2
        #if start is mentioned add the padding
        if sx!=-1 and sy !=-1:
            sx += pad
            sy += pad
        #if end is not mentioned default to width and height
        #else add padding to the end 
        if ex==-1 or ey ==-1:
            ex,ey = iW, iH
        else:
            ex += pad 
            ey += pad
        #convert the raw contour into its distance transform
        ip = self.geodesic_distance(ip, ksize=ksize, sx= sx, sy = sy, ex = ex, ey = ey)
        
        #perform the padding
        ip = cv2.copyMakeBorder(ip, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        for x in np.arange(sx, ex):
            for y in np.arange(sy, ey):
                if x+kW>iW or y+kH>iH:
                    #dirty fix to avoid cases where we overrun the array
                    continue
                roi = ip[x:x+kW,y:y+kH]
                #specific logic for region maxima 8-connected
                try:
                    if roi[roi.shape[0]/2, roi.shape[1]/2]>0:
                        maxval = np.max(roi)
                        if roi[kW/2,kH/2] == maxval:
                            _x = x - pad + kW/2
                            _y = y - pad + kH/2
                            output[_x, _y] = 1
                except:
                    print roi, x, x+kW, y, y+kH
                    return roi
        return output
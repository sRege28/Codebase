'''
Created on May 1, 2017

@author: arvind
'''

import numpy as np
import cv2
import georasters as gr
from matplotlib import pyplot as plt
import gdal
from spectral import principal_components as pc
import shapefile
from osgeo import osr
from shapely.geometry import Polygon

class BaseModel(object):
    '''
    classdocs
    '''
    

    def __init__(self, plotno):
        '''
        Constructor
        '''
        self.plotno = plotno
        self.load_data()
    
    def load_data(self):
        '''
            b
            g
            r
            img
            img_bgr
            img_gray
            img_pc
            chm
            shapes
        '''
        pwd = '/home/arvind/MyStuff/Coursework/PDS/NIST-DATA/NIST_data_20170120/'
        #loading image
        filepath = pwd + 'raster/camera/OSBS_' + self.plotno + '_camera.tif'
        camera_file = gdal.Open(filepath)
        self.b = np.flipud(camera_file.GetRasterBand(1).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
        self.g = np.flipud(camera_file.GetRasterBand(2).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
        self.r = np.flipud(camera_file.GetRasterBand(3).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
        self.img = cv2.merge([self.r,self.g,self.b])
        self.img_bgr = cv2.merge([self.b,self.g,self.r])
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        self.img_dims = (self.img.shape[0], self.img.shape[0])
        
        #loading hs
        filepath = pwd + 'raster/hs_fullBands/OSBS_' + self.plotno + '_nm350_2512.tif'
        hsimg = gr.from_file(filepath)
        hsimg = cv2.resize(hsimg.raster.T, self.img_dims)
        hsimg = np.rot90(hsimg, 3, (0,1))
        hsimg = np.fliplr(hsimg)
        hsimg = np.flipud(hsimg)
        pca_transform = pc(hsimg)
        pca_three_channel = pca_transform.reduce(num=3)
        img_pc = pca_three_channel.transform(hsimg).astype('float64')
        self.img_fc = img_pc[:,:,0]
        self.img_fc_gray = cv2.cvtColor(img_pc.astype('float32'), cv2.COLOR_BGR2GRAY)
        
        #loading chm
        filepath = pwd + 'raster/chm/OSBS_' + self.plotno + '_chm.tif'
        chmimg = gr.from_file(filepath)
        chm = chmimg.raster.data
        self.chmimg = chmimg
        chm = cv2.resize(chm, self.img_dims)
        self.chm = np.flipud(chm)
        
        #loading projection
        dataset = gdal.Open(filepath)
        sr = dataset.GetProjectionRef()
        osrobj = osr.SpatialReference()
        osrobj.ImportFromWkt(sr)
        srs = osr.SpatialReference()
        srs.ImportFromWkt('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        self.ct = osr.CoordinateTransformation( osrobj, srs )
        self.ct1 = osr.CoordinateTransformation( srs, osrobj )
        
        
        #loading shape files
        sf = shapefile.Reader(pwd + 'vector/Final Untagged Trees.shp')
        shapes = sf.shapes()
        sf = shapefile.Reader(pwd + 'vector/Final Tagged Trees.shp')
        self.shapes = shapes + sf.shapes()
        
    def imgcp(self):
        return np.copy(self.img)
    
    def execute(self):
        raise NotImplementedError
        
    
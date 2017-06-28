'''
Created on May 1, 2017

@author: arvind
'''

from jinja2 import Environment, FileSystemLoader
import datetime
import numpy as np
from weasyprint import HTML
import cv2
import georasters as gr
from matplotlib import pyplot as plt
import gdal
import shapefile
from osgeo import osr
import pylab as py
from os import listdir
from os.path import isfile, join
from eval_itcd_task import DelineationMetric
from eval_align_task import AlignMetric
from eval_classify_task import ClassifyMetric

class ReportGenerator(object):
    '''
    classdocs
    '''
    
    def __init__(self, config):
        '''
        Constructor
        '''
        params = config.template_vars
        date = datetime.datetime.now()
        params['time_submitted'] = str(date)
        self.template_vars = params
        pwd = config.datadir
        sf = shapefile.Reader(pwd + 'vector/Final Untagged Trees.shp')
        shapes = sf.shapes()
        sf = shapefile.Reader(pwd + 'vector/Final Tagged Trees.shp')
        self.shapes = shapes + sf.shapes()
        self.number_of_plots = 42
        self.config = config
        self.is_debug = False
        self.resolution_of_graphs = 100
        
    def generate(self):
        report_template = self.config.template_dir
        env = Environment(loader=FileSystemLoader(report_template))
        template = env.get_template('DSEReportTemplate.html')
        self.generate_task_1()
        self.generate_task_2()
        self.generate_task_3()
        date = datetime.datetime.now()
        self.template_vars['time_evaluated'] = str(date)
        html_out = template.render(self.template_vars)
        HTML(string=html_out, base_url=report_template).write_pdf(self.config.outdir + 'report.pdf', zoom=1.0, stylesheets=["template/style.css"])
        print "Completed successfully."
        
    def generate_task_3(self):
        task3_evaluator = ClassifyMetric(self.config)
        self.template_vars['t3_score'] = '%.3f' % (task3_evaluator.evaluate()* 100)
        self.template_vars['t3_r1_score'] = '%.3f' % (task3_evaluator.rank_1_acc * 100)
        precision_map = task3_evaluator.get_precision_map()
        self.plot_and_save(precision_map, #map
                           'Species', #xlabel
                           'Precision', #ylabel 
                           'Species Classification Precision',  #title
                           'species_classification_precision.png'); #filename
        
        
        recall_map = task3_evaluator.get_recall_map()
        self.plot_and_save(recall_map, #map
                           'Species', #xlabel
                           'Recall', #ylabel 
                           'Species Classification Recall',  #title
                           'species_classification_recall.png'); #filename
        self.draw_confusion_matrix_table(task3_evaluator.confusion_matrix, task3_evaluator.species_list)
        self.template_vars['confusion_matrix_table'] = task3_evaluator.confusion_matrix
        self.template_vars['species_list']  = task3_evaluator.species_list
        
    def intersects(self, r1, r2):
        return (r1[0] < r2[2] and r1[2] > r2[0] and r1[1] < r2[3] and r1[3] > r2[1] ) 
    
    def generate_task_1(self):
        pwd  = self.config.indir
        bpwd = self.config.datadir
        task1_evaluator = DelineationMetric()
        files = [f for f in listdir(pwd) if isfile(join(pwd, f)) and f.endswith('shp')]
        self.number_of_plots = len(files)
        
        for f in files:
            sf_pred = shapefile.Reader(pwd+f)

            
            plotno = f.split('_')[1].split('.')[0]
            filepath =  bpwd + 'raster/chm/OSBS_' + plotno + '_chm.tif'
            chmimg = gr.from_file(filepath)
            
            #loading image
            filepath = bpwd + 'raster/camera/OSBS_' + plotno + '_camera.tif'
            camera_file = gdal.Open(filepath)
            b = np.flipud(camera_file.GetRasterBand(1).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
            g = np.flipud(camera_file.GetRasterBand(2).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
            r = np.flipud(camera_file.GetRasterBand(3).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
            img = cv2.merge([r,g,b])
            
            sf = shapefile.Reader(bpwd+'/vector/Final Tagged Trees.shp')
            #reading projection extent from chmimg
            plot_extent = [chmimg.xmin,chmimg.ymin,chmimg.xmax,chmimg.ymax]
            dataset = gdal.Open(filepath)
            sr = dataset.GetProjectionRef()
            osrobj = osr.SpatialReference()
            osrobj.ImportFromWkt(sr)
            srs = osr.SpatialReference()
            srs.ImportFromWkt('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
            ct = osr.CoordinateTransformation( osrobj, srs )
            ct1 = osr.CoordinateTransformation( srs, osrobj )
            _plot = ct.TransformPoint(plot_extent[0], plot_extent[1])
            _plot = _plot[0:2] + ct.TransformPoint(plot_extent[2], plot_extent[3])
            _plot = _plot[:-1]
            _plot = np.array(_plot)
            shp_cont = []
            
            # checks for polygons which intersect with the current region of interest(plot)
            # also converts world coordinates to pixel coordinates to have a standard form 
            # which enables jaccard computation. Since the transform is affine it shouldnt matter!
            
            for shape in sf.shapes():
                if self.intersects(_plot, shape.bbox):
                    points = []
                    for p in shape.points:
                        l = ct1.TransformPoint(p[0], p[1])[:-1] - np.array(plot_extent[:2])
                        l[0] = (l[0] * img.shape[0])/(plot_extent[2] - plot_extent[0])
                        l[1] = (l[1] * img.shape[1])/(plot_extent[3] - plot_extent[1])
                        points.append(np.int32(np.ceil(l)))
                    shp_cont.append(np.array([points]))
            sf = shapefile.Reader(bpwd+'/vector/Final Untagged Trees.shp')
            for shape in sf.shapes():
                if self.intersects(_plot, shape.bbox):
                    points = []
                    for p in shape.points:
                        l = ct1.TransformPoint(p[0], p[1])[:-1] - np.array(plot_extent[:2])
                        l[0] = (l[0] * img.shape[0])/(plot_extent[2] - plot_extent[0])
                        l[1] = (l[1] * img.shape[1])/(plot_extent[3] - plot_extent[1])
                        points.append(np.int32(np.ceil(l)))
                    shp_cont.append(np.array([points]))
            shp_pred = []
            for shape in sf_pred.shapes():
                points = []
                for p in shape.points:
                    points.append(np.int32(p))
                shp_pred.append(np.array([points]))
           
            _, base, pred = task1_evaluator.calculateHungarianAssignment(plotno, shp_cont, shp_pred)
            if self.is_debug and plotno=='006':
                img1 = img.copy()
                for pair in task1_evaluator.assigmentMap:
                    cv2.drawContours(img1, [base[pair[0]]], -1, thickness=1, color=[0,255,0])
                    cv2.drawContours(img1, [pred[pair[1]]], -1, thickness=1, color=[255,0,0])
                plt.imshow(img1)
                plt.show()
                
        i = 0
        bottom5, top5 = task1_evaluator.getTopPolygons()
        for entry in top5:
            plotno = entry[0][0]
            contour = entry[1][1]
            self.draw_contour_and_save(plotno, contour, 'top5', i)
            i+=1
    
        i = 0
        for entry in bottom5:
            plotno = entry[0][0]
            contour = entry[1][1]
            self.draw_contour_and_save(plotno, contour, 'bottom5', i)
            i+=1
            
        
        ind = np.arange(self.number_of_plots)   
        width = 0.2
        
        p1 = plt.bar(ind, task1_evaluator.plotLevelTruePositives, width, color='g')
        p2 = plt.bar(ind, task1_evaluator.plotLevelFalsePositives, width, color='r', bottom=task1_evaluator.plotLevelTruePositives)
        p3 = plt.bar(ind, task1_evaluator.plotLevelFalseNegatives, width, color='b', bottom=task1_evaluator.plotLevelFalsePositives)
    
        plt.ylabel('Scores')
        plt.title('Plot Level Confusion Matrix')
        plt.ylim(1,np.max([np.max(task1_evaluator.plotLevelTruePositives), np.max(task1_evaluator.plotLevelFalsePositives), np.max(task1_evaluator.plotLevelFalseNegatives)]))
        plt.legend((p1[0], p2[0], p3[0]), ('TruePositive', 'FalsePositive','FalseNegative'))
        py.savefig(self.config.outdir + 'confusionMatrix.png', bbox_inches='tight', dpi=self.resolution_of_graphs)
        
        self.template_vars['t1_score'] = '%.3f' % (task1_evaluator.getFinalJaccardScore())
        plt.clf()
        task1_evaluator.getHistogramForRecall()
        py.savefig(self.config.outdir + 'histogramMatrix.png', bbox_inches='tight', dpi=self.resolution_of_graphs)
        tp, fp, fn = task1_evaluator.getConfusionMatrix()
        self.template_vars['true_positive'] = tp
        self.template_vars['false_positive'] = fp
        self.template_vars['true_negative'] = '-'
        self.template_vars['false_negative'] = fn


    def generate_task_2(self):
        task2_evaluator = AlignMetric(self.config)
        self.template_vars['t2_score'] = '%.3f' % (task2_evaluator.evaluate()* 100)
        count_correct_pred = task2_evaluator.plotwise_accuracy
        self.plot_and_save(count_correct_pred, #map
                           'Plot No.', #xlabel
                           'Count of Correct Alignment', #ylabel 
                           'Crown Alignment Accuracy',  #title
                           'crown_alignment.eps'); #filename
    
    def plot_and_save(self, val_map, xlab, ylab, title, filename):
        plt.clf()
        bar_width = 1
        plt.bar(range(len(val_map)), val_map.values(), width=bar_width)
        # _, labels = plt.xticks(range(len(val_map)), val_map.keys())
        # plt.setp(labels, rotation=90)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        py.savefig(self.config.outdir + filename)
    
    def draw_contour_and_save(self, plotno, contour, filepfx, count):
        filepath = self.config.datadir + 'raster/camera/OSBS_' + plotno + '_camera.tif'
        camera_file = gdal.Open(filepath)        
        col = [0,0,255]
        if filepfx.find('top')!=-1:
            col = [0, 255, 0]
        b = np.flipud(camera_file.GetRasterBand(1).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
        g = np.flipud(camera_file.GetRasterBand(2).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
        r = np.flipud(camera_file.GetRasterBand(3).ReadAsArray(0, 0, camera_file.RasterXSize, camera_file.RasterYSize).astype(np.uint8))
        img = cv2.merge([r,g,b])
        cv2.drawContours(img, [contour], -1, color=col, thickness=2)
        cv2.imwrite(self.config.outdir + filepfx + '_'+str(count)+'.jpg', img)
        
    def draw_confusion_matrix_table(self, confusion_matrix, species_list):
        plt.clf()
        _, axs =plt.subplots(1,1)
        col_width=.070
        axs.axis('tight')
        axs.axis('off')
        tab = axs.table(cellText=np.int32(confusion_matrix),loc='center')
        tab.auto_set_font_size(False) 
        tab.set_fontsize(10) 
        tab.scale(1.2, 1.5)

        hoffset=-0.07
        voffset=1.12 
        count=0
        for s in species_list:
            axs.annotate('  '+s , xy=(hoffset+count*col_width,voffset),
                xycoords='axes fraction', ha='left', va='bottom', 
                rotation=90, size=10)
            count+=1
        py.savefig(self.config.outdir + 'confusion_matrix_table.eps', bbox_inches='tight', dpi=self.resolution_of_graphs)

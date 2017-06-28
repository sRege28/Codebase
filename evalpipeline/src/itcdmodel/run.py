import logmodel as LM
import shapefile
import numpy as np
for i in range(51,52):
        try:
            print 'omg %03d' % i
            polys = LM.LoGModel('%03d' % i).execute()   
         
            w = shapefile.Writer(shapeType=5)
            w.field("plotnumber", "C")
            j = 0
            for p in polys:
                j+=1 
                w.poly(parts=np.array(p).reshape(len(p),1,2).tolist())                
                w.record("%03d" % i)
            w.save('/home/arvind/shpout/shp_%03d.shp' % i)
        except:
            pass


import operator
import processing
import timeit

tic=timeit.default_timer() # Timer begins

min_lhd = 3.799824611

hyp_grid_layer = qgis.utils.iface.mapCanvas().layers()[0]
exist_grid_areas = qgis.utils.iface.mapCanvas().layers()[1]

multigeom = QgsGeometry.fromWkt("GEOMETRYCOLLECTION")
for f in exist_grid_areas.getFeatures():
    multigeom = multigeom.combine(f.geometry())

grids_list = []
for f in hyp_grid_layer.selectedFeatures():
    d = f.geometry().distance(multigeom)
    grids_list.append([f, d, f['lhd_mwh_m']])
grids_list.sort(key = operator.itemgetter(1,2))

curr_length = 1208740.0 #Meter
curr_heat_demand = 4593000.0 #MWh

out_feats = []
for g in grids_list:
    
    
    supply_line = g[0].geometry().shortestLine(multigeom) #find distance again, since the geometry will be changed
    d = supply_line.length()
    new_l =  g[0]['length'] + d
    new_demand = g[0]['w_erz_total'] / 1000
    new_dens = new_demand/new_l
    
    print new_l, new_demand, new_dens, g[0].id()
    
    if new_dens > min_lhd:
        
        print "connect"
        
        if d != 0:
            g[0].setGeometry(g[0].geometry().combine(supply_line))
        
        g[0].setAttribute('length', new_l)
        g[0].setAttribute('lhd_mwh_m', new_dens)
        
        out_feats.append(g[0])
        curr_length += g[0]['length'] + g[1]
        curr_heat_demand += g[0]['w_erz_total'] / 1000
        multigeom = multigeom.combine(g[0].geometry().buffer(1,3))

    else:
        pass

out_layer = QgsVectorLayer("MultiLineString", "temp_layer", "memory")
out_layer.dataProvider().addAttributes(hyp_grid_layer.fields())
out_layer.updateFields()
out_layer.dataProvider().addFeatures(out_feats)
QgsMapLayerRegistry.instance().addMapLayer(out_layer)
error = QgsVectorFileWriter.writeAsVectorFormat(out_layer, "S:\HiDrive\Ivan\District Heating Conference Hamburg 2018\Grids_GEWISS_builds_X_VDI_grown_from_gridT", "utf-8", None, "SQLite", False, None, ['SPATIALITE=YES', ]) #ESRI Shapefile PGDump PostGIS
if error == QgsVectorFileWriter.NoError:
    print "grids success!"

toc=timeit.default_timer()
print "Time : " + str((toc - tic)/60)

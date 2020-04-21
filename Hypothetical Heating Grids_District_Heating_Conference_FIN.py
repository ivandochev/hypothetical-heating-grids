#Version 28.2.2018. MST function adopted from Andreas Mueller:
#http://peekaboo-vision.blogspot.de/2012/02/simplistic-minimum-spanning-tree-in.html
#It requires ALKIS, addresses and Clusters

import math 
import timeit
import itertools
import operator
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import groupby
from PyQt4.QtCore import QVariant

outpath = "S:\HiDrive\Ivan\District Heating Conference Hamburg 2018\Grids_GEWISS_builds_X_VDI_MFH_grid.sqlite"

#PREPARE FUNCTIONS
def MST(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
     
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges)

#remove non-terminals of degree 1 recursively
def rem_non_terminals(edge_list, non_terms):

    #print edge_list
    #print "---"
    #check number of points with degree==1
    def check_degree(edge_list, non_terms):
        count = 0
        degree_vals = []
        for steiner in non_terms:
            degree = list(edge_list.flatten()).count(steiner)
            degree_vals.append(degree)
            if degree == 1:
                count+=1
            else:
                pass
        return count
    
    cnt = check_degree(edge_list, non_terms)
    
    while cnt != 0:
        for steiner in non_terms:
            degree = list(edge_list.flatten()).count(steiner)
            if degree == 1:
                rows, cols = np.where(edge_list == steiner)
                edge_list = np.delete(edge_list,rows,0)
            else:
                pass
        cnt = check_degree(edge_list, non_terms)
        #print cnt
    
    
    #print edge_list
    #print non_terms
    return edge_list
        #print steiner
        #print list(edge_list.flatten()).count(steiner)
        #print np.where(edge_list != steiner)

def MSTPLine(edge_list,points, build_multigeom):
    geoms = QgsGeometry.fromWkt('GEOMETRYCOLLECTION()')
    
    for edge in edge_list:
        i, j = edge
        p1 = QgsPoint(points[i, 0], points[i, 1])
        p2 = QgsPoint(points[j, 0], points[j, 1])
        
        #line = get_around_builds(p1,p2,build_multigeom)
        line = QgsGeometry().fromPolyline([p1,p2])
        geoms = geoms.combine(line)
    
    return geoms

def densify(line,dist):
    L = line.length()
    currentDist = dist
    points = []
    while currentDist < L:
        point = line.interpolate(currentDist).asPoint()
        points.append(point)
        currentDist += dist
    return points


#BEGIN
tic=timeit.default_timer() # Timer begins
cluster_field = 'cluster'
gemarking_f = 'gemarkung'
flurstueck_f = 'flurstueck'
adr_field_buildings = 'address'
adr_field_addresses = 'address'
zeigtAuf = 'zeigtAuf'
heat_demand = 'w_erz_total'

#Out Fields
fields = QgsFields()
fields.append(QgsField("w_erz_total", QVariant.Int))
fields.append(QgsField("str_id", QVariant.String))
fields.append(QgsField("length", QVariant.Double))
fields.append(QgsField("LHD_MWh_m", QVariant.Double)) 

#PREPARE DICTIONARIES
tic=timeit.default_timer() # Timer begins
canvas = qgis.utils.iface.mapCanvas()
layers = canvas.layers()
#Layer order in the canvas becomes important!!!
street_dict = {s.id(): s for s in layers[0].getFeatures()}
adr_point_dict = {a[adr_field_addresses]: a for a in layers[1].getFeatures()}
build_dict = {b.id(): b for b in layers[2].selectedFeatures()}

toc=timeit.default_timer()
print "Prepair Dics : " + str((toc - tic)/60)

#PREPARE STREET INDEX
street_sp_index = QgsSpatialIndex(layers[0].getFeatures())

#0. PREPARE BUILDNG LIST
feature_list = []
for f in build_dict.values():
    #if f[cluster_field] != "Anonymized" and f[cluster_field] != "clus":
    if f["w_erz_total"] != 0 and f.geometry().area() > 30:
        feature_list.append([
                    "%s_%s" % (f[gemarking_f], f[flurstueck_f]), #Here I changed to match the plot level, to avoid referencing the clustering, it is a different story
                    f
                    ])


#GROUP BASED ON CLUSTER
feature_list = sorted(feature_list, key=operator.itemgetter(0))
clusters = []
for cl, clgroup in groupby(feature_list, lambda x: x[0]):
    clusters.append(list(clgroup))


#FIND THE NEAREST STREET TO EACH CLUSTER
for cl in clusters:
    #print cl
    multigeom = QgsGeometry.fromWkt('GEOMETRYCOLLECTION()')
    
    for b in cl:
        multigeom = multigeom.combine(b[1].geometry().buffer(0,3))
        
    near_streets = street_sp_index.nearestNeighbor(multigeom.centroid().asPoint(),5)
        
    in_dist = 10000
    for n in near_streets:
        n_geom = street_dict[n].geometry()
        dist = n_geom.distance(multigeom.centroid())
        if dist < in_dist:
            nearest_id = n
            in_dist = dist
            
    
    for b in cl:
        b.append(nearest_id)

toc=timeit.default_timer()
print "Find nearest Street : " + str((toc - tic)/60)

#ADD HEAT TRANSFER STATION GEOMETRY
#This is the approximated geographic location of the heat transfer station

feature_list = [b for cl in clusters for b in cl]
for f in feature_list:
    #take the  first address
    if f[1][zeigtAuf] != NULL:
        address1 = f[1][adr_field_buildings].split(",")[0]
        try:
            adr_point = adr_point_dict[address1].geometry()
        except KeyError:
            adr_point = f[1].geometry().centroid()
    else:
        adr_point = f[1].geometry().centroid()
    
    build_outer_ring = f[1].geometry().convexHull().buffer(1.5,3).convertToType(QGis.Line)
    transfer_st_point = build_outer_ring.nearestPoint(adr_point)
    adr_st_connect = QgsGeometry().fromPolyline([transfer_st_point.asPoint(), adr_point.asPoint()])
    
    
    f.append(transfer_st_point)
    f.append("tr_station")
    f.append(adr_st_connect)

toc=timeit.default_timer()
print "Heat Transfer Station Geometry : " + str((toc - tic)/60)

#CREATE GRID_CLUSTERS
#resort
feature_list = sorted(feature_list, key=operator.itemgetter(2))
#create grid_clusters
grid_clusters = []
for grid_cl, grid_clgroup in groupby(feature_list, lambda x: x[2]):
    grid_clusters.append(list(grid_clgroup))

#[u'212033_5',  Cluster Name
#<qgis._core.QgsFeature object at 0x00000000B53656A8>,  Build Feature
#7245L,  Street ID
#<qgis._core.QgsGeometry object at 0x000000009E3CD598>] Transfer Station Coordinates
#Type "street", "bbuffer", "tr_station"
#the last segment between the adr_p and the connection point outside the buildings 


#CREATE GRID GEOMETRY
#Define spatial relation between transfer stations
grids_feature_list = []
for grid_cluster in grid_clusters:
    #Get the whole cluster as geometry and the data
    multigeom = QgsGeometry.fromWkt('GEOMETRYCOLLECTION()')
    adr_st_connect = QgsGeometry.fromWkt('GEOMETRYCOLLECTION()') #the last small bits between the adr_p and the tr.station
    total_demand = 0
    str_id = grid_cluster[0][2]
    for b in grid_cluster:
        multigeom = multigeom.combine(b[1].geometry().buffer(0,0))
        total_demand += b[1][heat_demand]
        adr_st_connect = adr_st_connect.combine(b[5])

    
    #append the street points
    str_geom = street_dict[grid_cluster[0][2]].geometry()
    str_vertices = []
    #str_points = densify(str_geom,2)
    for st in grid_cluster:
        str_vertices.append([None, None, None, str_geom.nearestPoint(st[3]), "street", None])
    for line in str_geom.asGeometryCollection():
        for vrtx in line.asPolyline():
            str_vertices.append([None, None, None, QgsGeometry().fromPoint(vrtx), "street", None])
    
    grid_cluster += str_vertices

    toc=timeit.default_timer()
    #print "Append Street Points : " + str((toc - tic)/60)
    
    #get the ids of the corner points)  from
    from_index = len(grid_cluster)
    
    '''
    #Get the corner points
    build_buffer_points = []
    #get fewer Steiner points by merging a bit of the geometries
    multigeom = multigeom.buffer(10,1,2,2,2).buffer(-10,1,2,2,2)
    
    for merged_b in multigeom.asGeometryCollection():
        area = merged_b.area()
        simplified_buffer = merged_b.buffer(1.5,1,2,2,2).simplify(math.sqrt(area)/10)
        print simplified_buffer.exportToWkt()
        for corner_point in simplified_buffer.asPolygon()[0]:
            build_buffer_points.append([None, None, None, QgsGeometry().fromPoint(corner_point), "bbuffer"])
    
    grid_cluster += build_buffer_points 
    '''
    toc=timeit.default_timer()
    #print "Get corner points : " + str((toc - tic)/60)
    
    #get the ids of the corner points)  to
    to_index = len(grid_cluster)
    non_terms = range(from_index, to_index)
    
    #for p in rounded:
        #print QgsGeometry().fromPoint(p).exportToWkt()

    #Find distances between polygon geoms with GEOS distance and create complete graph
    dists = []
    for pair in itertools.combinations(grid_cluster, 2):
        p1 = pair[0][3]
        p2 = pair[1][3]
        
        distance = p1.distance(p2)
        
        #get around buildings
        #get_around_builds(p1.asPoint(),p2.asPoint(),multigeom)
        
        #DEFINE THE CONNECTION RULES
        line = QgsGeometry().fromPolyline([p1.asPoint(),p2.asPoint()])
        #if line.intersects(multigeom):
            #distance = distance #!!!!!!

        if pair[0][4] == pair[1][4] == 'street':
            distance = distance*0.1
        
        #Different plot (cluster)
        elif pair[0][0] != None and pair[1][0] != None and pair[0][0] != pair[1][0]:
            distance = distance*2
        
        dists.append(distance)
        
    #get array in square form
    points = np.array([f[3].asPoint() for f in grid_cluster])
    dists = np.array(dists)
    X = squareform(dists)
    
    toc=timeit.default_timer()
    #print "Prep points and combinations : " + str((toc - tic)/60)

    #Compute edges and geometry
    edge_list = MST(X)
    toc=timeit.default_timer()
    #print "Compute MST : " + str((toc - tic)/60)
    
    rem_non_terms = rem_non_terminals(edge_list, non_terms)
    toc=timeit.default_timer()
    #print "Remove Non Terminals : " + str((toc - tic)/60)
    
    grid_geom = MSTPLine(rem_non_terms,points, multigeom)
    grid_geom = grid_geom.combine(adr_st_connect)
    
    grid_length = grid_geom.length()
    LHD = total_demand/(1000*grid_length)
    toc=timeit.default_timer()
    #print "Get Geometry : " + str((toc - tic)/60)
    
    #prep feature
    feat = QgsFeature(fields)
    feat.setGeometry(grid_geom)
    feat.setAttribute('w_erz_total', total_demand)
    feat.setAttribute('str_id', str_id)
    feat.setAttribute('length', grid_length)
    feat.setAttribute('LHD_MWh_m', LHD)
    grids_feature_list.append(feat)



grid_layer = QgsVectorLayer("MultiLineString", "grids", "memory")
grid_layer.dataProvider().addAttributes(fields) #[2] is QgsFields
grid_layer.updateFields()
grid_layer.dataProvider().addFeatures(grids_feature_list)
grid_layer.isValid()

error = QgsVectorFileWriter.writeAsVectorFormat(grid_layer, outpath, "utf-8", None, "SQLite", False, None, ['SPATIALITE=YES', ]) #ESRI Shapefile PGDump PostGIS
if error == QgsVectorFileWriter.NoError:
    print "grids success!"

toc=timeit.default_timer()
print "Minutes elapsed : " + str((toc - tic)/60)

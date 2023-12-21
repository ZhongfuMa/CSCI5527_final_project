import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
from datetime import datetime
from geopy.distance import geodesic
from tqdm import tqdm
import numpy as np
from matplotlib.ticker import PercentFormatter
from shapely import geometry

def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

# print(os.getcwd())

def aggragate_data(path):
    HEADER=['device_key',
            'unix_time',
            'min_duration_seconds',
            'observation_count',
            'cluster_type',
            'visit_centroid_lat',
            'visit_centroid_lon',
            'visit_dispersion',
            'trajectory_start_lat',
            'trajectory_start_lon',
            'trajectory_end_lat',
            'trajectory_end_lon',
            'movement_source_key',
            'dma_code',
            'state_abbr',
            'zipcode_5']

    d=[]
#     counter=0
    for file_path in path:
#         counter+=1
#         print(file_path)
        df=pd.read_csv(file_path,usecols=range(1,len(HEADER)+1))
        df.columns=HEADER
        d.append(df)

    d1=pd.concat(d) # whole day 2021/03/01
    d1.reset_index(inplace=True)
    d1.drop('index',axis=1,inplace=True)
    d1['date_time'] = [datetime.fromtimestamp(x) for x in d1['unix_time']]
    date = d1['date_time']
    d1.drop(labels=['date_time'], axis=1, inplace = True)
    d1.insert(1, 'date_time', date)
    d1['date_time'] = pd.to_datetime(d1['date_time'])
    
    return d1

def mkdir(path):
 
    folder = os.path.exists(path)
 
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
        
        
def create_fishnet(TCMA,cell_size):
    # Create fishnet
    # Get the extent of the shapefile
    total_bounds = TCMA.total_bounds

    # Get minX, minY, maxX, maxY
    minX, minY, maxX, maxY = total_bounds

    # Create a fishnet
    x, y = (minX, minY)
    geom_array = []

    # Polygon Size
    square_size = cell_size
    while y <= maxY:
        while x <= maxX:
            geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
            geom_array.append(geom)
            x += square_size
        x = minX
        y += square_size

    fishnet = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs('EPSG:26915')

    ax=fishnet.plot(color='white', edgecolor='black',figsize=(10,10))
    TCMA.plot(ax=ax)
    plt.title(str(cell_size)+'m')
    plt.show()

    intersections=fishnet.intersects(TCMA.unary_union)
    intersections=intersections[intersections==True]
    overlap_fishnet=fishnet.iloc[list(intersections.index),:]

    return overlap_fishnet

def process_org_file(f):
    
    global boundary
    global save_path
    
    try:
        start_time = time.time()
        HEADER=['device_key',
            'unix_time',
            'min_duration_seconds',
            'observation_count',
            'cluster_type',
            'visit_centroid_lat',
            'visit_centroid_lon',
            'visit_dispersion',
            'trajectory_start_lat',
            'trajectory_start_lon',
            'trajectory_end_lat',
            'trajectory_end_lon',
            'movement_source_key',
            'dma_code',
            'state_abbr',
            'zipcode_5']
        df=pd.read_csv(f,header=None)#,header=None
        df.columns=HEADER

        d1_tra=df.query('cluster_type=="trajectory"')
        gdf_d1_o= gpd.GeoDataFrame(d1_tra, geometry=gpd.points_from_xy(d1_tra.trajectory_start_lon, d1_tra.trajectory_start_lat))

        d1_tra=df.query('cluster_type=="trajectory"')
        gdf_d1_d= gpd.GeoDataFrame(d1_tra, geometry=gpd.points_from_xy(d1_tra.trajectory_end_lon, d1_tra.trajectory_end_lat))

        gdf_d1_o.crs= "EPSG:4326" 
        gdf_d1_d.crs= "EPSG:4326" 

        gdf_d1_o=gdf_d1_o.to_crs(boundary.crs)
        gdf_d1_d=gdf_d1_d.to_crs(boundary.crs)

        o_with_boundary = gpd.sjoin(gdf_d1_o, boundary, how="inner")
        d_with_boundary = gpd.sjoin(gdf_d1_d, boundary, how="inner")

        o_with_boundary=o_with_boundary.rename(columns={'CensusBlockGroup':'boundary_o'})
        d_with_boundary=d_with_boundary.rename(columns={'CensusBlockGroup':'boundary_d'})

        od_boundary=pd.concat([o_with_boundary,d_with_boundary['boundary_d']],axis=1)
        od_boundary['od']=od_boundary['boundary_o']+','+od_boundary['boundary_d']

        od_boundary_clear=od_boundary[od_boundary['od'].isnull()==False]

        od_boundary_clear=od_boundary_clear[HEADER]

        f_element=f.split('/')
        od_boundary_clear.to_csv(save_path+'/'+f_element[-2]+'/'+f_element[-1].split('.')[0]+'.csv')
        end_time = time.time()
        print('Finish:'+f)
        print("cost: {:.2f} sec".format(end_time - start_time))
        
        return od_boundary_clear
    except BaseException:
        print('Error:'+f)
        return f
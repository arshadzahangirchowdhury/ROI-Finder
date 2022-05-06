#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
ROI-Finder function
"""

import sys
if '../' not in sys.path:
    sys.path.append('../')
    
import roifinder.src.roif_config
from roifinder.src.roif_config import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from matplotlib import rc
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import seaborn as sns

import roifinder.src.xrf_roif_internals

from roifinder.src.xrf_roif_internals import *

import roifinder.src.xrfm_batches
from roifinder.src.xrfm_batches import *

import roifinder.src.fuzzy_kmeans_engine
from roifinder.src.fuzzy_kmeans_engine import _format_axes, soft_clustering_weights

from sklearn import metrics

import cv2

import warnings
warnings.filterwarnings("ignore")

def ROI_Finder(
                base_file_path,
                coarse_scan_names,
                hdf5_string_list,
                norm_ch_list,
                selected_elm_maps_list,
                noise_type_list,
                bin_conv_elm_list,
                value_offset_list,
                apply_gaussian_list,
                BASE_PATCH_WIDTH,
                pixel_threshold=8,
                normalize = False,
                print_pv=False,  
                verbosity=False ,
                mode='single' 
                ):
    
    '''
    
    Function to train on multiple XRF images and on a target XRF image detect and return 
    all class 0 and class 1 ecoli cells/ROIs based on selected mode, the function speciefies which class refers to type A or type B cells.
    The target XRF file must be the last item in the list passed in the coarse scan names.
    
    args:
    base_file_path: string,
    coarse_scan_names: list,
    hdf5_string_list: list,
    norm_ch_list: list,
    selected_elm_maps_list: list,
    noise_type_list: list ,
    bin_conv_elm_list: list,
    value_offset_list: list,
    apply_gaussian_list:list,
    BASE_PATCH_WIDTH: int,
    pixel_threshold:int, cells with pixels below this threshold are considered artifacts. Default value is 8.
    normalize = False,
    print_pv=False,  
    verbosity=False ,
    mode: string, 'single' returns sorted motor coordinates of 1 cell from class 0 and 1 cell from class 1 based on confidence.
                   'all' sorted returns motor coordinates of a cells from class 0 and all cells from class 1 based on confidence.
                   'develop' returns a pandas dataframe consisting of all the information for desired processing.
                   
                   
    returns: motor_coordinate(s) based on x_res for BNP based on selected mode and the calculated confidence metric in the form of (x,y, confidence). 
    For 'single' or 'auto' modes, sorting is performed via confidence. For 'develop mode', the secondaryDf dataframe is returned               
    '''
    
    
    
    coarse_scans = XRFM_batch(base_file_path,
                  coarse_scan_names,
                 hdf5_string_list,
                 norm_ch_list,
                 selected_elm_maps_list,
                 noise_type_list,
                 bin_conv_elm_list,
                 value_offset_list,
                apply_gaussian_list,
                 BASE_PATCH_WIDTH,
                 print_pv=False,  
                 verbosity=False)

    print('Bounding box width and height (pixels):' , BASE_PATCH_WIDTH)
    print('Total extracted cells, features:', coarse_scans.X.shape)
    print('Total extracted cell, cell size:', coarse_scans.X_bin.shape)

    


    #------------------------

    principalDf = pd.DataFrame(
                 columns = ['Pixel_count', 'area'])


    principalDf['area'] = coarse_scans.X[:,0]
    principalDf['eccentricity'] = coarse_scans.X[:,1]
    principalDf['equivalent_diameter'] = coarse_scans.X[:,2]
    principalDf['major_axis_length'] = coarse_scans.X[:,3]
    principalDf['minor_axis_length'] = coarse_scans.X[:,4]
    principalDf['perimeter'] = coarse_scans.X[:,5]
    principalDf['K'] = coarse_scans.X[:,6]
    principalDf['P'] = coarse_scans.X[:,7]
    principalDf['Ca'] = coarse_scans.X[:,8]
    principalDf['Zn'] = coarse_scans.X[:,9]
    principalDf['Fe'] = coarse_scans.X[:,10]
    principalDf['Cu'] = coarse_scans.X[:,11]
    principalDf['BFY'] = coarse_scans.X[:,12]
    principalDf['Pixel_count'] = coarse_scans.X[:,13].astype(int)   #Pixel_count column must exist

    #add res and origins to dataframe here
    #convert from list
    principalDf['x_res'] = coarse_scans.X_x_res
    principalDf['y_res'] = coarse_scans.X_y_res
    principalDf['avg_res'] = coarse_scans.X_avg_res
    principalDf['x_origin'] = coarse_scans.X_x_origin
    principalDf['y_origin'] = coarse_scans.X_y_origin
    principalDf['x_motor'] = coarse_scans.X_x_motor
    principalDf['y_motor'] = coarse_scans.X_y_motor
    principalDf['xrf_file']=coarse_scans.X_xrf_track_files

    #assign scan names in dataframe
    number_of_cells = principalDf['xrf_file'].to_numpy().shape[0]
    coarse_scan_name=[]
    for idx in range(number_of_cells):

        coarse_scan_name.append(os.path.split(principalDf['xrf_file'].to_numpy()[idx])[1])
    principalDf['scan_name'] =  np.array(coarse_scan_name)  

    secondaryDf=remove_artifacts(principalDf, remove_count = pixel_threshold)

    
    

#     #-------PCA-kmeans-----------------

    mod_X = np.asarray([
    secondaryDf['area'],secondaryDf['eccentricity'],
    secondaryDf['K'],secondaryDf['P'],secondaryDf['Ca'],secondaryDf['Zn'],secondaryDf['Fe']
               ]).T

    print('Cells, features', mod_X.shape)
    
    
    X_standard = StandardScaler().fit_transform(mod_X)


    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_standard)

    #assign PC values to dataframe
    secondaryDf['PC1']=principalComponents[:,0]
    secondaryDf['PC2']=principalComponents[:,1]

    kmeans = KMeans(n_clusters = 2, random_state = 42).fit(secondaryDf[['PC1','PC2']].to_numpy())

    secondaryDf['Class'] = kmeans.labels_

    print('k-means assigned class to max K cell:', secondaryDf.iloc[secondaryDf['K'].idxmax()]['Class'])
    print('High K cell should be 1. Is it?', secondaryDf.iloc[secondaryDf['K'].idxmax()]['Class'] ==1 )
    print('High K cell is class ', secondaryDf.iloc[secondaryDf['K'].idxmax()]['Class'] )
    
    

    
    
    for i in range(3):
        secondaryDf['p' + str(i)] = 0
    secondaryDf[['p0', 'p1']] = soft_clustering_weights(principalComponents, kmeans.cluster_centers_)
    secondaryDf['confidence'] = np.max(secondaryDf[['p0', 'p1']].values, axis = 1)
    
    
    currentXRF_Df = secondaryDf[secondaryDf['xrf_file'] == secondaryDf['xrf_file'].iloc[-1]]
    
    class_0_array = currentXRF_Df[currentXRF_Df['Class']==0][['x_motor','y_motor', 'confidence']].to_numpy()
    class_1_array = currentXRF_Df[currentXRF_Df['Class']==1][['x_motor','y_motor', 'confidence']].to_numpy()
    
    
    
    
# #     display(secondaryDf)
    print('class_0 cells found:',class_0_array[:,0].size)
    print('class_1 cells found:',class_1_array[:,0].size)
    
    print('in scan: ',secondaryDf['xrf_file'].iloc[-1])
    
    if mode == 'auto':
        
    
        return class_0_array[class_0_array[:, 2].argsort()[::-1]], class_1_array[class_1_array[:, 2].argsort()[::-1]]
    
    elif mode == 'single':
        print('class 0 cell original idx:',currentXRF_Df[currentXRF_Df['Class']==0][['original index']].to_numpy()[0])
        print('class 1 cell original idx:',currentXRF_Df[currentXRF_Df['Class']==1][['original index']].to_numpy()[0])
        return class_0_array[class_0_array[:, 2].argsort()[::-1]][0], class_1_array[class_1_array[:, 2].argsort()[::-1]][0]
    
    elif mode == 'develop':
        print('class 0 cell original idx:',currentXRF_Df[currentXRF_Df['Class']==0][['original index']].to_numpy())
        print('class 1 cell original idx:',currentXRF_Df[currentXRF_Df['Class']==1][['original index']].to_numpy())
        return secondaryDf
    
    else:
        print('Select mode to be either auto or single. Returning location as mode auto without sorting')
        return class_0_array, class_1_array


if __name__ == "__main__":
    
    print('just ROI-finder functions!')
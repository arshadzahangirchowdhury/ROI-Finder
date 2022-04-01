#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury
Creates XRF datasets for use by various functions and subroutines

"""

import sys
import os
import numpy as np
import pandas as pd
import glob
import h5py
import time
import tifffile as tiff




class two_class_xrf_data():
    '''
    
    A class to handle the annotated data based on accept and not_a_cell classes.
    
    base_path : string, path location of the data
    base_accept_path : string, path location of the accepted cells
    base_not_cell_path: string, path location of the not_cells or background cells
    
    returns:
    accepts : list of the 9 channels of the XRF images deemed to be 'live' or 'accepted'.
    not_cells : list of the 9 channels of the XRF images deemed to be 'background' or 'not cells'.
    
    '''
    
    def __init__(self, 
                 base_path = '/data01/AZC/annotated_xrf_simple/', 
                 base_accept_path= '/data01/AZC/annotated_xrf_simple/accept/', 
                 base_not_cell_path='/data01/AZC/annotated_xrf_simple/not_cell/', 
                 BASE_PATCH_WIDTH=28,
                 verbosity = False):
        
        self.base_path = base_path
        self.base_accept_path = base_accept_path
        self.base_not_cell_path = base_not_cell_path
        self.base_path = base_path
        self.verbosity=verbosity
        
    
    
    
    def _load_xrf_tif_files(self,
                            class_type = 'accept',
                            exp_tag= 'bnp_fly0001_2018_1',
                            elm_channel = 'Ca'):
    
    
        path_to_tif= os.path.join(self.base_path,class_type,exp_tag,elm_channel, '*.tif')

#         print(path_to_tif)

        img_list = []

        for img in glob.glob(path_to_tif):
            n= tiff.imread(img)
            img_list.append(n)

        return np.array(img_list, dtype=np.float32)
    
    
    def load_data(self):
    # Single experiment
    # bnp_fly0001_2018_1
    # Elms = Cu, Zn, Ca, K, P, S, Fe, Ni and TFY
        
        list_exp_tags=['bnp_fly0001_2018_1',
              'bnp_fly0001_2018_3',
              'bnp_fly0003_2018_3',
              'bnp_fly0012_2018_1w2',
              'bnp_fly0014_2018_1w2',
              'bnp_fly0040_2018_1',
              'bnp_fly0050_2018_1',
              'bnp_fly0051_2018_1',
              'bnp_fly0052_2018_1',
              'bnp_fly0065_2018_3']
        
            
        # each variable contains a nested list of images
        exp_accept_n_Cu = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Cu') for x in list_exp_tags])

        exp_accept_n_Zn = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Zn') for x in list_exp_tags])

        exp_accept_n_Ca = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Ca') for x in list_exp_tags])

        exp_accept_n_K = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'K') for x in list_exp_tags])

        exp_accept_n_P = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'P') for x in list_exp_tags])

        exp_accept_n_S = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'S') for x in list_exp_tags])

        exp_accept_n_Fe = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Fe') for x in list_exp_tags])

        exp_accept_n_Ni = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Ni') for x in list_exp_tags])

        exp_accept_n_TFY = ([self._load_xrf_tif_files(class_type='accept', 
                                                     exp_tag = x , 
                                                     elm_channel = 'TFY') for x in list_exp_tags])

        exp_not_cell_n_Cu = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Cu') for x in list_exp_tags])

        exp_not_cell_n_Zn = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Zn') for x in list_exp_tags])

        exp_not_cell_n_Ca = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Ca') for x in list_exp_tags])

        exp_not_cell_n_K = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'K') for x in list_exp_tags])

        exp_not_cell_n_P = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'P') for x in list_exp_tags])

        exp_not_cell_n_S = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'S') for x in list_exp_tags])

        exp_not_cell_n_Fe = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Fe') for x in list_exp_tags])

        exp_not_cell_n_Ni = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'Ni') for x in list_exp_tags])

        exp_not_cell_n_TFY = ([self._load_xrf_tif_files(class_type='not_cell', 
                                                     exp_tag = x , 
                                                     elm_channel = 'TFY') for x in list_exp_tags])
        #print(exp_accept_n_Cu.shape)

        accept_Cu = np.concatenate(exp_accept_n_Cu)
        accept_Zn = np.concatenate(exp_accept_n_Zn)
        accept_Ca = np.concatenate(exp_accept_n_Ca)
        accept_K = np.concatenate(exp_accept_n_K)
        accept_P = np.concatenate(exp_accept_n_P)
        accept_S = np.concatenate(exp_accept_n_S)
        accept_Fe = np.concatenate(exp_accept_n_Fe)
        accept_Ni = np.concatenate(exp_accept_n_Ni)
        accept_TFY = np.concatenate(exp_accept_n_TFY)
        
        not_cell_Cu = np.concatenate(exp_not_cell_n_Cu)
        not_cell_Zn = np.concatenate(exp_not_cell_n_Zn)
        not_cell_Ca = np.concatenate(exp_not_cell_n_Ca)
        not_cell_K = np.concatenate(exp_not_cell_n_K)
        not_cell_P = np.concatenate(exp_not_cell_n_P)
        not_cell_S = np.concatenate(exp_not_cell_n_S)
        not_cell_Fe = np.concatenate(exp_not_cell_n_Fe)
        not_cell_Ni = np.concatenate(exp_not_cell_n_Ni)
        not_cell_TFY = np.concatenate(exp_not_cell_n_TFY)
        
        
        accepts = [accept_Cu, accept_Zn,accept_Ca, accept_K, accept_P, accept_S, accept_Fe, accept_Ni,accept_TFY,]
        not_cells = [not_cell_Cu, not_cell_Zn, not_cell_Ca, not_cell_K, not_cell_P, not_cell_S, not_cell_Fe, not_cell_Ni, not_cell_TFY]
        
        return accepts, not_cells
    
    
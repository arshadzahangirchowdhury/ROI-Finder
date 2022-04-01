#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Loads 2018 XRF data for ecoli bacteria. This data can be used to train PCA,k-means, supervised neural nets

'''

import beamtime_config_2018
import ROI_Finder_2018
from beamtime_config_2018 import *
from ROI_Finder_2018 import *

noise_type = 'none'
base_file_path='/data02/AZC/XRF_Data/'

def load_XRF_dataset_2018(noise_type = 'none', base_file_path='/data01/AZC/XRF_Data/' ):
    
    '''
    loads 2018 XRF data, converts to binary and extract patches based on most relevant conversion parameters found.
    
    Arguments:
    ----------
    noise_type: string, 'poisson' argunment will add noise defned in ROI-Finder_2018.
    base_file_path: string, path to folder containing XRF data
    
    Returns:
    --------
    data_dict:dict, contains handmade features
    
    '''
    
    #file 1

    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0001_2018_1.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='Cu')
    x.extract_cells()
    X_bin1=x.binary_img  


    X_Cu1=x.Patches_Cu
    X_Zn1=x.Patches_Zn
    X_Ca1=x.Patches_Ca
    X_K1=x.Patches_K
    X_P1=x.Patches_P
    X_S1=x.Patches_S
    X_Fe1=x.Patches_Fe
    X_Ni1=x.Patches_Ni
    X_TFY1=x.Patches_TFY
    
    X_x_res1=x.x_res_list
    X_y_res1=x.y_res_list
    X_avg_res1=x.avg_res_list
    X_x_origin1=x.x_origin_list
    X_y_origin1=x.y_origin_list
    
    X_x_motor1=x.x_motor_center_list
    X_y_motor1=x.y_motor_center_list


    SNR_X1=calc_SNR(x.d_Cu, x.binary_ero_dil)

    X1=x.features
    
    X_centers1=x.center_coords
    X_xrf_track_file1=x.XRF_track_files
    
    X1.shape

    #file 2


    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0001_2018_3.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)
    x.binary_conversion(e='K')
    x.extract_cells()
    X_bin2=x.binary_img

    X_Cu2=x.Patches_Cu
    X_Zn2=x.Patches_Zn
    X_Ca2=x.Patches_Ca
    X_K2=x.Patches_K
    X_P2=x.Patches_P
    X_S2=x.Patches_S
    X_Fe2=x.Patches_Fe
    X_Ni2=x.Patches_Ni
    X_TFY2=x.Patches_TFY
    
    X_x_res2=x.x_res_list
    X_y_res2=x.y_res_list
    X_avg_res2=x.avg_res_list
    X_x_origin2=x.x_origin_list
    X_y_origin2=x.y_origin_list
    
    X_x_motor2=x.x_motor_center_list
    X_y_motor2=x.y_motor_center_list



    SNR_X2=calc_SNR(x.d_K, x.binary_ero_dil)

    X2=x.features
    
    X_centers2=x.center_coords
    X_xrf_track_file2=x.XRF_track_files
    
    X2.shape

    #file 3

    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0003_2018_3.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='K')
    x.extract_cells()
    X_bin3=x.binary_img

    X_Cu3=x.Patches_Cu
    X_Zn3=x.Patches_Zn
    X_Ca3=x.Patches_Ca
    X_K3=x.Patches_K
    X_P3=x.Patches_P
    X_S3=x.Patches_S
    X_Fe3=x.Patches_Fe
    X_Ni3=x.Patches_Ni
    X_TFY3=x.Patches_TFY
    
    X_x_res3=x.x_res_list
    X_y_res3=x.y_res_list
    X_avg_res3=x.avg_res_list
    X_x_origin3=x.x_origin_list
    X_y_origin3=x.y_origin_list
    
    X_x_motor3=x.x_motor_center_list
    X_y_motor3=x.y_motor_center_list



    SNR_X3=calc_SNR(x.d_K, x.binary_ero_dil)

    X3=x.features
    
    X_centers3=x.center_coords
    X_xrf_track_file3=x.XRF_track_files
    
    X3.shape

    #file 4

    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0012_2018_1w2.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='Cu')
    x.extract_cells()
    X_bin4=x.binary_img

    X_Cu4=x.Patches_Cu
    X_Zn4=x.Patches_Zn
    X_Ca4=x.Patches_Ca
    X_K4=x.Patches_K
    X_P4=x.Patches_P
    X_S4=x.Patches_S
    X_Fe4=x.Patches_Fe
    X_Ni4=x.Patches_Ni
    X_TFY4=x.Patches_TFY
    
    X_x_res4=x.x_res_list
    X_y_res4=x.y_res_list
    X_avg_res4=x.avg_res_list
    X_x_origin4=x.x_origin_list
    X_y_origin4=x.y_origin_list
    
    X_x_motor4=x.x_motor_center_list
    X_y_motor4=x.y_motor_center_list


    SNR_X4=calc_SNR(x.d_Cu, x.binary_ero_dil)

    X4=x.features
    
    X_centers4=x.center_coords
    X_xrf_track_file4=x.XRF_track_files
    
    X4.shape

    #file 5

    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0014_2018_1w2.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='Cu')
    x.extract_cells()

    X_bin5=x.binary_img

    X_Cu5=x.Patches_Cu
    X_Zn5=x.Patches_Zn
    X_Ca5=x.Patches_Ca
    X_K5=x.Patches_K
    X_P5=x.Patches_P
    X_S5=x.Patches_S
    X_Fe5=x.Patches_Fe
    X_Ni5=x.Patches_Ni
    X_TFY5=x.Patches_TFY
    
    X_x_res5=x.x_res_list
    X_y_res5=x.y_res_list
    X_avg_res5=x.avg_res_list
    X_x_origin5=x.x_origin_list
    X_y_origin5=x.y_origin_list
    
    X_x_motor5=x.x_motor_center_list
    X_y_motor5=x.y_motor_center_list


    SNR_X5=calc_SNR(x.d_Cu, x.binary_ero_dil)


    X5=x.features
    
    X_centers5=x.center_coords
    X_xrf_track_file5=x.XRF_track_files
    
    X5.shape

    #file 6

    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0040_2018_1.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='Cu')
    x.extract_cells()
    X_bin6=x.binary_img

    X_Cu6=x.Patches_Cu
    X_Zn6=x.Patches_Zn
    X_Ca6=x.Patches_Ca
    X_K6=x.Patches_K
    X_P6=x.Patches_P
    X_S6=x.Patches_S
    X_Fe6=x.Patches_Fe
    X_Ni6=x.Patches_Ni
    X_TFY6=x.Patches_TFY
    
    X_x_res6=x.x_res_list
    X_y_res6=x.y_res_list
    X_avg_res6=x.avg_res_list
    X_x_origin6=x.x_origin_list
    X_y_origin6=x.y_origin_list
    
    X_x_motor6=x.x_motor_center_list
    X_y_motor6=x.y_motor_center_list


    SNR_X6=calc_SNR(x.d_Cu, x.binary_ero_dil)

    X6=x.features
    
    X_centers6=x.center_coords
    X_xrf_track_file6=x.XRF_track_files
    
    X6.shape

    #file 7

    x= XRF_image(base_file_path +'/bnp_fly0050_2018_1.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='Cu')
    x.extract_cells()
    X_bin7=x.binary_img

    X_Cu7=x.Patches_Cu
    X_Zn7=x.Patches_Zn
    X_Ca7=x.Patches_Ca
    X_K7=x.Patches_K
    X_P7=x.Patches_P
    X_S7=x.Patches_S
    X_Fe7=x.Patches_Fe
    X_Ni7=x.Patches_Ni
    X_TFY7=x.Patches_TFY
    
    X_x_res7=x.x_res_list
    X_y_res7=x.y_res_list
    X_avg_res7=x.avg_res_list
    X_x_origin7=x.x_origin_list
    X_y_origin7=x.y_origin_list
    
    X_x_motor7=x.x_motor_center_list
    X_y_motor7=x.y_motor_center_list


    SNR_X7=calc_SNR(x.d_Cu, x.binary_ero_dil)

    X7=x.features
    
    X_centers7=x.center_coords
    X_xrf_track_file7=x.XRF_track_files
    
    X7.shape

    #file 8

    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0051_2018_1.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='Cu')
    x.extract_cells()
    X_bin8=x.binary_img


    X_Cu8=x.Patches_Cu
    X_Zn8=x.Patches_Zn
    X_Ca8=x.Patches_Ca
    X_K8=x.Patches_K
    X_P8=x.Patches_P
    X_S8=x.Patches_S
    X_Fe8=x.Patches_Fe
    X_Ni8=x.Patches_Ni
    X_TFY8=x.Patches_TFY
    
    X_x_res8=x.x_res_list
    X_y_res8=x.y_res_list
    X_avg_res8=x.avg_res_list
    X_x_origin8=x.x_origin_list
    X_y_origin8=x.y_origin_list
    
    X_x_motor8=x.x_motor_center_list
    X_y_motor8=x.y_motor_center_list


    SNR_X8=calc_SNR(x.d_Cu, x.binary_ero_dil)

    X8=x.features
    
    X_centers8=x.center_coords
    X_xrf_track_file8=x.XRF_track_files
    
    X8.shape

    # file 9

    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0052_2018_1.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)

    x.binary_conversion(e='Cu')
    x.extract_cells()
    X_bin9=x.binary_img

    X_Cu9=x.Patches_Cu
    X_Zn9=x.Patches_Zn
    X_Ca9=x.Patches_Ca
    X_K9=x.Patches_K
    X_P9=x.Patches_P
    X_S9=x.Patches_S
    X_Fe9=x.Patches_Fe
    X_Ni9=x.Patches_Ni
    X_TFY9=x.Patches_TFY
    
    X_x_res9=x.x_res_list
    X_y_res9=x.y_res_list
    X_avg_res9=x.avg_res_list
    X_x_origin9=x.x_origin_list
    X_y_origin9=x.y_origin_list
    
    X_x_motor9=x.x_motor_center_list
    X_y_motor9=x.y_motor_center_list


    SNR_X9=calc_SNR(x.d_Cu, x.binary_ero_dil)

    X9=x.features
    X_centers9=x.center_coords
    X_xrf_track_file9=x.XRF_track_files
    
    X9.shape

    #file 10
    x= XRF_image(xrf_filename = base_file_path + 'bnp_fly0065_2018_3.h5',
                 BASE_PATCH_WIDTH=32, verbosity=False)
    x.load_xrf_data(hdf5_string='exchange_4')
    x.add_noise(noise=noise_type)
    x.binary_conversion(e='K')
    x.extract_cells()
    X_bin10=x.binary_img

    X_Cu10=x.Patches_Cu
    X_Zn10=x.Patches_Zn
    X_Ca10=x.Patches_Ca
    X_K10=x.Patches_K
    X_P10=x.Patches_P
    X_S10=x.Patches_S
    X_Fe10=x.Patches_Fe
    X_Ni10=x.Patches_Ni
    X_TFY10=x.Patches_TFY
    
    X_x_res10=x.x_res_list
    X_y_res10=x.y_res_list
    X_avg_res10=x.avg_res_list
    X_x_origin10=x.x_origin_list
    X_y_origin10=x.y_origin_list
    
    X_x_motor10=x.x_motor_center_list
    X_y_motor10=x.y_motor_center_list



    SNR_X10=calc_SNR(x.d_K, x.binary_ero_dil)

    X10=x.features
    
    X_centers10=x.center_coords
    X_xrf_track_file10=x.XRF_track_files

    X10.shape

    X=np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10))
    X_bin=np.concatenate((X_bin1,X_bin2,X_bin3,X_bin4,X_bin5,X_bin6,X_bin7,X_bin8,X_bin9,X_bin10))
    X_Cu=np.concatenate((X_Cu1,X_Cu2,X_Cu3,X_Cu4,X_Cu5,X_Cu6,X_Cu7,X_Cu8,X_Cu9,X_Cu10))
    X_Zn=np.concatenate((X_Zn1,X_Zn2,X_Zn3,X_Zn4,X_Zn5,X_Zn6,X_Zn7,X_Zn8,X_Zn9,X_Zn10))
    X_Ca=np.concatenate((X_Ca1,X_Ca2,X_Ca3,X_Ca4,X_Ca5,X_Ca6,X_Ca7,X_Ca8,X_Ca9,X_Ca10))
    X_K=np.concatenate((X_K1,X_K2,X_K3,X_K4,X_K5,X_K6,X_K7,X_K8,X_K9,X_K10))
    X_P=np.concatenate((X_P1,X_P2,X_P3,X_P4,X_P5,X_P6,X_P7,X_P8,X_P9,X_P10))
    X_S=np.concatenate((X_S1,X_S2,X_S3,X_S4,X_S5,X_S6,X_S7,X_S8,X_S9,X_S10))
    X_Fe=np.concatenate((X_Fe1,X_Fe2,X_Fe3,X_Fe4,X_Fe5,X_Fe6,X_Fe7,X_Fe8,X_Fe9,X_Fe10))
    X_Ni=np.concatenate((X_Ni1,X_Ni2,X_Ni3,X_Ni4,X_Ni5,X_Ni6,X_Ni7,X_Ni8,X_Ni9,X_Ni10))
    X_TFY=np.concatenate((X_TFY1,X_TFY2,X_TFY3,X_TFY4,X_TFY5,X_TFY6,X_TFY7,X_TFY8,X_TFY9,X_TFY10))
    
    X_x_res=np.concatenate((X_x_res1,X_x_res2,X_x_res3,X_x_res4,X_x_res5,X_x_res6,X_x_res7,X_x_res8,X_x_res9,X_x_res10))
    X_y_res=np.concatenate((X_y_res1,X_y_res2,X_y_res3,X_y_res4,X_y_res5,X_y_res6,X_y_res7,X_y_res8,X_y_res9,X_y_res10))
    X_avg_res=np.concatenate((X_avg_res1,X_avg_res2,X_avg_res3,X_avg_res4,X_avg_res5,X_avg_res6,X_avg_res7,X_avg_res8,X_avg_res9,X_avg_res10))
    
    X_x_origin=np.concatenate((X_x_origin1,X_x_origin2,X_x_origin3,X_x_origin4,X_x_origin5,X_x_origin6,X_x_origin7,X_x_origin8,X_x_origin9,X_x_origin10))
    X_y_origin=np.concatenate((X_y_origin1,X_y_origin2,X_y_origin3,X_y_origin4,X_y_origin5,X_y_origin6,X_y_origin7,X_y_origin8,X_y_origin9,X_y_origin10))
    
    
    X_x_motor=np.concatenate((X_x_motor1,X_x_motor2,X_x_motor3,X_x_motor4,X_x_motor5,X_x_motor6,X_x_motor7,X_x_motor8,X_x_motor9,X_x_motor10))
    X_y_motor=np.concatenate((X_y_motor1,X_y_motor2,X_y_motor3,X_y_motor4,X_y_motor5,X_y_motor6,X_y_motor7,X_y_motor8,X_y_motor9,X_y_motor10))
 
    

    
    X_centers=np.concatenate((X_centers1,X_centers2,X_centers3,X_centers4,X_centers5,
                              X_centers6,X_centers7,X_centers8,X_centers9,X_centers10))

    X_xrf_track_files=np.concatenate((X_xrf_track_file1,X_xrf_track_file2,X_xrf_track_file3,X_xrf_track_file4,X_xrf_track_file5,
                      X_xrf_track_file6,X_xrf_track_file7,X_xrf_track_file8,X_xrf_track_file9,X_xrf_track_file10))


    SNR=np.array([SNR_X1,SNR_X2,SNR_X3,SNR_X4,SNR_X5,SNR_X6,SNR_X7,SNR_X8,SNR_X9,SNR_X10])
    SNR_Df = pd.DataFrame()
    SNR_Df['SNR']=SNR

    # print(SNR_Df.to_string())

    print('Total extracted cells, features:', X.shape)
    print('Total extracted cell, cell size:', X_bin.shape)

    principalDf = pd.DataFrame(
                 columns = ['Pixel_count', 'area'])

    principalDf['area'] = X[:,0]
    principalDf['eccentricity'] = X[:,1]
    principalDf['equivalent_diameter'] = X[:,2]
    principalDf['major_axis_length'] = X[:,3]
    principalDf['minor_axis_length'] = X[:,4]
    principalDf['perimeter'] = X[:,5]
    principalDf['K'] = X[:,6]
    principalDf['P'] = X[:,7]
    principalDf['Ni'] = X[:,8]
    principalDf['Zn'] = X[:,9]
    principalDf['Fe'] = X[:,10]
    principalDf['Cu'] = X[:,11]
    principalDf['BFY'] = X[:,12]
    principalDf['Pixel_count'] = X[:,13].astype(int)
    # display(principalDf)
    
    
    #add res and origins to dataframe here
    
    principalDf['x_res'] = X_x_res
    principalDf['y_res'] = X_y_res
    principalDf['avg_res'] = X_avg_res
    principalDf['x_origin'] = X_x_origin
    principalDf['y_origin'] = X_y_origin
    principalDf['x_motor'] = X_x_motor
    principalDf['y_motor'] = X_y_motor

    
    
    annotation_file_path=base_file_path + 'xrf_annotations_arshad_KO_full_indices.csv'
    KO_annotations = pd.read_csv(annotation_file_path)

    principalDf['KO_label']=KO_annotations['KO_label'].to_numpy()
    principalDf['K_obs_labels']=KO_annotations['K'].to_numpy()
    principalDf['Ni_obs_labels']=KO_annotations['Ni'].to_numpy()
    principalDf['Fe_obs_labels']=KO_annotations['Fe'].to_numpy()
    principalDf['Cu_obs_labels']=KO_annotations['Cu'].to_numpy()
    principalDf['Ca_obs_labels']=KO_annotations['Ca'].to_numpy()
    principalDf['Zn_obs_labels']=KO_annotations['Zn'].to_numpy()

    principalDf.KO_label = principalDf.KO_label.fillna('not_annotated')
    principalDf.K_obs_labels = principalDf.K_obs_labels.fillna('not_annotated')
    principalDf.Ni_obs_labels = principalDf.Ni_obs_labels.fillna('not_annotated')
    principalDf.Fe_obs_labels = principalDf.Fe_obs_labels.fillna('not_annotated')
    principalDf.Cu_obs_labels = principalDf.Cu_obs_labels.fillna('not_annotated')
    principalDf.Ca_obs_labels = principalDf.Ca_obs_labels.fillna('not_annotated')
    principalDf.Zn_obs_labels = principalDf.Zn_obs_labels.fillna('not_annotated')    
    
    
    data_dict = {'X':X, 'X_bin':X_bin, 'X_Cu':X_Cu, 'X_Zn':X_Zn, 'X_Ca':X_Ca, 'X_K':X_K,
                 'X_P':X_P,'X_S':X_S,'X_Fe':X_Fe,'X_Ni':X_Ni, 'X_TFY':X_TFY, 'principalDf':principalDf,
                'X_centers':X_centers,'X_xrf_track_files':X_xrf_track_files, 
                 'x_res':X_x_res, 'y_res':X_y_res,'avg_res':X_avg_res,'x_origin':X_x_origin,'y_origin':X_y_origin,
                 'x_motor':X_x_motor,'y_motor':X_y_motor,
                'BASE_PATCH_WIDTH':x.BASE_PATCH_WIDTH}
    
    return data_dict
    
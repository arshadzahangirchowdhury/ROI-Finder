
import beamtime_config_2022
from beamtime_config_2022 import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from matplotlib import rc
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import seaborn as sns

import xrf_roif_2022

from xrf_roif_2022 import *


class XRFM_batch:
    '''
    Creates a batch of beamtime_XRF_image class takes assemble and organize all coarse scans for ROI-Finder.
    
    arguments:
    base_file_path,
    coarse_scan_names,
    hdf5_string_list,
    norm_ch_list,
    selected_elm_maps_list,
    noise_type_list,
    bin_conv_elm_list,
    value_offset_list,
    BASE_PATCH_WIDTH,
    print_pv=False,  
    verbosity=False 
    
    attribute
    
    '''
    
    def __init__(self,
                 base_file_path,
                 coarse_scan_names,
                 hdf5_string_list,
                 norm_ch_list,
                 selected_elm_maps_list,
                 noise_type_list,
                 bin_conv_elm_list,
                 value_offset_list,
                 BASE_PATCH_WIDTH,
                 print_pv=False,  
                 verbosity=False ):
        
        self.base_file_path = base_file_path
        self.coarse_scan_names = coarse_scan_names
        self.hdf5_string_list = hdf5_string_list
        self.norm_ch_list = norm_ch_list
        self.selected_elm_maps_list = selected_elm_maps_list
        self.noise_type_list = noise_type_list
        self.bin_conv_elm_list = bin_conv_elm_list
        self.value_offset_list=value_offset_list
        self.BASE_PATCH_WIDTH=BASE_PATCH_WIDTH
        self.verbosity=verbosity
        self.print_pv=print_pv
        
        #image channels
        self.X_d_Cu=[]
        self.X_d_Zn=[]
        self.X_d_Ca=[]
        self.X_d_K=[]
        self.X_d_P=[]
        self.X_d_S=[]
        self.X_d_Fe=[]
        self.X_d_Ni=[]
        self.X_d_TFY=[]
        self.X_binary_ero_dil=[]
        
        #cell images
            
        self.X_bin=[]  
        self.X_Cu=[]
        self.X_Zn=[]
        self.X_Ca=[]
        self.X_K=[]
        self.X_P=[]
        self.X_S=[]
        self.X_Fe=[]
        self.X_Ni=[]
        self.X_TFY=[]

        self.X_x_res=[]
        self.X_y_res=[]
        self.X_avg_res=[]
        self.X_x_origin=[]
        self.X_y_origin=[]

        self.X_x_motor=[]
        self.X_y_motor=[]


        self.SNR_X=[]

        self.X=[]

        self.X_centers=[]
        self.X_xrf_track_files=[]


        

        for (a, b, c, d, e, f,g) in zip(self.coarse_scan_names, self.hdf5_string_list, self.selected_elm_maps_list, 
                                        self.noise_type_list, self.bin_conv_elm_list,self.norm_ch_list,self.value_offset_list):
        #     print (a, b, c, d, e, f,g)
            # x is a single coarse scan image
            x= beamtime_XRF_image(xrf_filename = self.base_file_path + a,
                     BASE_PATCH_WIDTH=self.BASE_PATCH_WIDTH, norm_ch=f,value_offset=g,print_pv=self.print_pv, verbosity=self.verbosity)
            x.load_xrf_data(hdf5_string=b)
            x.load_element_maps(selected_elm_maps = c)
            x.add_noise(noise=d)

            x.binary_conversion(e=e)
            x.extract_cells()
            x.define_features(mode='max')
            
            #image channels
            
            self.X_d_Cu.append(x.d_Cu)
            self.X_d_Zn.append(x.d_Zn)
            self.X_d_Ca.append(x.d_Ca)
            self.X_d_K.append(x.d_K)
            self.X_d_P.append(x.d_P)
            self.X_d_S.append(x.d_S)
            self.X_d_Fe.append(x.d_Fe)
            self.X_d_Ni.append(x.d_Ni)
            self.X_d_TFY.append(x.d_TFY)
            self.X_binary_ero_dil.append(x.binary_ero_dil)

            #cell images
            self.X_bin.append(x.binary_img)
            self.X_Cu.append(x.Patches_Cu)
            self.X_Zn.append(x.Patches_Zn)
            self.X_Ca.append(x.Patches_Ca)
            self.X_K.append(x.Patches_K)
            self.X_P.append(x.Patches_P)
            self.X_S.append(x.Patches_S)
            self.X_Fe.append(x.Patches_Fe)
            self.X_Ni.append(x.Patches_Ni)
            self.X_TFY.append(x.Patches_TFY)

            self.X_x_res.append(x.x_res_list)
            self.X_y_res.append(x.y_res_list)
            self.X_avg_res.append(x.avg_res_list)
            self.X_x_origin.append(x.x_origin_list)
            self.X_y_origin.append(x.y_origin_list)

            self.X_x_motor.append(x.x_motor_center_list)
            self.X_y_motor.append(x.y_motor_center_list)
            
            #calculate SNR

            if e == 'Cu':
                self.SNR_X.append(calc_SNR(x.d_Cu, x.binary_ero_dil))

            if e == 'Zn':
                self.SNR_X.append(calc_SNR(x.d_Zn, x.binary_ero_dil))

            if e == 'Ca':
                self.SNR_X.append(calc_SNR(x.d_Ca, x.binary_ero_dil))

            if e == 'K':
                self.SNR_X.append(calc_SNR(x.d_K, x.binary_ero_dil))

            if e == 'P':
                self.SNR_X.append(calc_SNR(x.d_P, x.binary_ero_dil))

            if e == 'S':
                self.SNR_X.append(calc_SNR(x.d_S, x.binary_ero_dil))

            if e == 'Fe':
                self.SNR_X.append(calc_SNR(x.d_Fe, x.binary_ero_dil))

            if e == 'Ni':
                self.SNR_X.append(calc_SNR(x.d_Ni, x.binary_ero_dil))

            if e == 'Total_Fluorescence_Yield' or e == 'TFY':
                self.SNR_X.append(calc_SNR(x.d_TFY, x.binary_ero_dil))




            self.X.append(x.features)

            self.X_centers.append(x.center_coords)
            self.X_xrf_track_files.append(x.XRF_track_files)

            print(x.features.shape)

        # combine all extractions to arrays
        
        self.X=np.vstack(self.X)
        self.X_bin=np.vstack(self.X_bin)
        self.X_Cu=np.vstack(self.X_Cu)
        self.X_Zn=np.vstack(self.X_Zn)
        self.X_Ca=np.vstack(self.X_Ca)
        self.X_K=np.vstack(self.X_K)
        self.X_P=np.vstack(self.X_P)
        self.X_S=np.vstack(self.X_S)
        self.X_Fe=np.vstack(self.X_Fe)
        self.X_Ni=np.vstack(self.X_Ni)
        self.X_TFY=np.vstack(self.X_TFY)

        self.X_x_res=np.concatenate([item for item in self.X_x_res], 0)
        self.X_y_res=np.concatenate([item for item in self.X_y_res], 0)
        self.X_avg_res=np.concatenate([item for item in self.X_avg_res], 0)

        self.X_x_origin=np.concatenate([item for item in self.X_x_origin], 0)
        self.X_y_origin=np.concatenate([item for item in self.X_y_origin], 0)

        self.X_x_motor=np.concatenate([item for item in self.X_x_motor], 0)
        self.X_y_motor=np.concatenate([item for item in self.X_y_motor], 0)



        # SNR=np.concatenate(SNR_X)
        # # SNR_Df = pd.DataFrame()
        # # SNR_Df['SNR']=SNR
        # # print(SNR_Df.to_string())

        self.X_centers=np.vstack(self.X_centers)

        self.X_xrf_track_files=np.concatenate([item for item in self.X_xrf_track_files], 0)


    def plot_coarse_binary_images(self):
        '''
        plots the binary images of the coarse scans
        '''

        for item in self.X_binary_ero_dil:
            print(item.shape)
            plt.figure(dpi=200)
            plt.imshow(item)
            plt.colorbar(orientation='horizontal', shrink=0.5)
            plt.title(self.coarse_scan_names[0])
            plt.gca().invert_yaxis()   
        

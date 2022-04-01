import beamtime_config_2022

from beamtime_config_2022 import *


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from matplotlib import rc
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import seaborn as sns
import plotly.express as px





class beamtime_XRF_image:
    '''
    beamtime_XRF_image class takes coarse scans as inuput
    
    arguments:
    xrf_filename
    BASE_PATCH_WIDTH
    norm_ch
    value_offset
    verbosity
    print_pv
    
    '''
    
    def __init__(self, xrf_filename = '/path_to .h5 file',
             BASE_PATCH_WIDTH=32, norm_ch = 'US_IC', value_offset = 1e-12,  verbosity=False, print_pv=False):
        self.xrf_filename =xrf_filename
        self.BASE_PATCH_WIDTH=BASE_PATCH_WIDTH
        self.verbosity=verbosity
        self.print_pv=print_pv
        self.norm_ch = norm_ch
        self.value_offset=value_offset
        
        
        
    def load_xrf_data(self, hdf5_string = 'MAPS'):
#         global d_Cu, d_Zn, d_Ca, d_K, d_P, d_S,d_Fe, d_Ni, d_TFY
    
#         norm_ch = NORM_CH # 2018 value
        print('XRF', self.xrf_filename)
#         norm_ch = 'US_IC'
#         value_offset=VALUE_OFFSET
        xrfdata = collections.defaultdict(list)

        with h5py.File(self.xrf_filename, 'r') as dat:
            
            
            if self.print_pv ==True:
                self.pvlist = dat["MAPS/extra_pvs"][0].astype(str).tolist();
                self.pvval = dat["MAPS/extra_pvs"][1].astype(str).tolist();


                print('pvlist:', self.pvlist)
                print('pvval:', self.pvval)

                if '9idbTAU:SM:SZ:ActPos' in self.pvlist:

                    self.pvindex = self.pvlist.index('9idbTAU:SM:SZ:ActPos')
                    print('pvindex:', self.pvindex)
                    print('pvval:', self.pvval[self.pvindex])
            
            
            groups= list(dat.keys())
            maps= list(dat['MAPS'].keys())
            chs = dat['MAPS/channel_names'][:].astype(str).tolist()
            
            if self.verbosity ==True:
                print('groups:', groups)
                print('maps:', maps)
                print('chs:', chs)
            
    
            

        #         dat['MAPS/']
            
            
            
#             self.int_spec=dat['MAPS/int_spec'][:].astype(int).tolist()
#             self.energy=dat['MAPS/energy'][:].astype(int).tolist()
            
#             self.mca_arr=dat['MAPS/mca_arr'][:].astype(int).tolist()
#             self.scan_time_stamp=dat['MAPS/scan_time_stamp']
#             print(self.scan_time_stamp)
            xrf = dat['MAPS/XRF_roi'][:]
        #         print(xrf)

            scaler_names = dat['MAPS/scaler_names'][:].astype(str).tolist()
#             print('scaler_names:', scaler_names)
            scaler_val = dat['MAPS/scalers'][:]
#             print(scaler_val)
            norm = scaler_val[scaler_names.index(self.norm_ch)]
#             print('norm:', norm)
            for e in chs:
                chidx = chs.index(e)
                xrfdata[e].append(xrf[chidx]/norm)
            xrfdata['scan_num'].append(self.xrf_filename)

            hdf5_string = hdf5_string

            xrfdata['x_axis'].append(dat[hdf5_string + '/x_axis'][:])
            xrfdata['y_axis'].append(dat[hdf5_string + '/y_axis'][:])
            
            
            
            

        
        
        xrfdata = pd.DataFrame(xrfdata)
        self.xrfdata=xrfdata
        
        if self.verbosity ==True:
            print('XRF info:',xrfdata.columns)
        
        
        
    def load_element_maps(self, selected_elm_maps=['Cu','Zn','Ca', 'K', 'P', 'S','Fe','Ni','Total_Fluorescence_Yield']):
#         elms=['Cu','Zn','Ca', 'K', 'P', 'S','Fe','Total_Fluorescence_Yield']#Default elms
        elms=selected_elm_maps #Default elms
        print(selected_elm_maps)
        for i, row in self.xrfdata.iterrows():
                sc = row['scan_num'][0:row['scan_num'].index('.')]
                for e in elms:
                    d = row[e]

                    d[np.isnan(d) | np.isinf(d)] = 0
                    norm_d = (d - np.min(d)) / (np.max(d) - np.min(d)) + self.value_offset
                    ss = np.round(np.abs(np.diff(row['x_axis']))[0], 2)
                    if e == 'Cu':
                        self.d_Cu=d
                        self.norm_d_Cu=norm_d
                        self.x_Cu,self.y_Cu=row['x_axis'], row['y_axis']
                    if e == 'Zn':
                        self.d_Zn=d
                        self.norm_d_Zn=norm_d
                        self.x_Zn,self.y_Zn=row['x_axis'], row['y_axis']
                    if e == 'Ca':
                        self.d_Ca=d
                        self.norm_d_Ca=norm_d
                        self.x_Ca,self.y_Ca=row['x_axis'], row['y_axis']
                    if e == 'K':
                        self.d_K=d
                        self.norm_d_K=norm_d
                        self.x_K,self.y_K=row['x_axis'], row['y_axis']
                    if e == 'P':
                        self.d_P=d
                        self.norm_d_P=norm_d
                        self.x_P,self.y_P=row['x_axis'], row['y_axis']
                    if e == 'S':
                        self.d_S=d
                        self.norm_d_S=norm_d
                        self.x_S,self.y_S=row['x_axis'], row['y_axis']

                    if e == 'Fe':
                        self.d_Fe=d
                        self.norm_d_Fe=norm_d
                        self.x_Fe,self.y_Fe=row['x_axis'], row['y_axis']

                    if e == 'Ni':
                        
                        self.d_Ni=d
                        self.norm_d_Ni=norm_d
                        self.x_Ni,self.y_Ni=row['x_axis'], row['y_axis']

                    if e == 'Total_Fluorescence_Yield' or e == 'TFY':
                        self.d_TFY=d
                        self.norm_d_TFY=norm_d
                        self.x_TFY,self.y_TFY=row['x_axis'], row['y_axis']
#             print('Image shape: ',d.shape)
            
        
        #motor coordinate, steps, resolution
        
        self.x_res=self.x_TFY[1]-self.x_TFY[0]
        self.y_res=self.y_TFY[1]-self.y_TFY[0]
        self.avg_res=(self.x_res+self.y_res)/2
        self.x_origin=self.x_TFY[0]
        self.y_origin=self.y_TFY[0]
        
        
        #debug info
#         print('x_res:',self.x_res)
#         print('y_res:',self.y_res)
#         print('avg_res:',self.avg_res)
        
#         print('x_origin:',self.x_origin)
#         print('y_origin:',self.y_origin)
        
    def normalize_XRFM(self, channel = 'S'):
        '''
        normalize each channels of the coarse scan with the channel.
        
        returns
        attributes holding the main scans for different elemental maps are normalized.
        
        '''
        
        if channel == 'S':
            
            self.d_Cu=self.d_Cu/np.amax(self.d_S)
            self.d_Zn=self.d_Zn/np.amax(self.d_S)
            self.d_Ca=self.d_Ca/np.amax(self.d_S)
            self.d_K=self.d_K/np.amax(self.d_S)
            self.d_P=self.d_P/np.amax(self.d_S)
            self.d_S=self.d_S/np.amax(self.d_S)
            self.d_Fe=self.d_Fe/np.amax(self.d_S)
            self.d_Ni=self.d_Ni/np.amax(self.d_S)
            self.d_TFY=self.d_TFY/np.amax(self.d_S)
            
            
            

        
    
    def add_noise(self, noise='none'):
        self.noise=noise
        if self.noise == 'normal':
            np.random.seed(0)
            self.normal_noise=abs(np.random.normal(0, 1, self.d_Cu.shape))
            self.d_Cu = self.d_Cu*(1+self.normal_noise) 
            self.d_Zn = self.d_Zn*(1+self.normal_noise)
            self.d_Ca = self.d_Ca*(1+self.normal_noise)
            self.d_K = self.d_K*(1+self.normal_noise)
            self.d_P = self.d_P*(1+self.normal_noise)
            self.d_S = self.d_S*(1+self.normal_noise)
            self.d_Fe = self.d_Fe*(1+self.normal_noise)
            self.d_Ni = self.d_Ni*(1+self.normal_noise)
            self.d_TFY = self.d_TFY*(1+self.normal_noise)
            
        elif self.noise == 'poisson':
            np.random.seed(0)
            self.poisson_noise=abs(np.random.poisson(1000, self.d_Cu.shape))
            self.d_Cu = self.d_Cu*(1+self.poisson_noise) 
            self.d_Zn = self.d_Zn*(1+self.poisson_noise)
            self.d_Ca = self.d_Ca*(1+self.poisson_noise)
            self.d_K = self.d_K*(1+self.poisson_noise)
            self.d_P = self.d_P*(1+self.poisson_noise)
            self.d_S = self.d_S*(1+self.poisson_noise)
            self.d_Fe = self.d_Fe*(1+self.poisson_noise)
            self.d_Ni = self.d_Ni*(1+self.poisson_noise)
            self.d_TFY = self.d_TFY*(1+self.poisson_noise)


        
    
    
    def binary_conversion(self, e='Cu'):
        #choose elemental channel for conversion
        
        if e == 'Cu':
            data_original = self.d_Cu
            
        if e == 'Zn':
            data_original = self.d_Zn
            
        if e == 'Ca':
            data_original = self.d_Ca
            
        if e == 'K':
            data_original = self.d_K
            
        if e == 'P':
            data_original = self.d_P
            
        if e == 'S':
            data_original = self.d_S
            

        if e == 'Fe':
            data_original = self.d_Fe
            

        if e == 'Cl':
            data_original = self.d_Ni
            
        if e == 'Total_Fluorescence_Yield':
            data_original = self.d_TFY
            
#         data_original=d_Cu
        data=data_original
        data = ndimage.median_filter(data, size=3)


        thresh = 1.25*threshold_otsu(data)
        binary = data < thresh
        binary = binary^1


        binary_eroded=ndimage.binary_erosion(binary).astype(binary.dtype)
        binary_dilated=ndimage.binary_dilation(binary).astype(binary.dtype)
        self.binary_ero_dil=ndimage.binary_dilation(binary_eroded).astype(binary_eroded.dtype)
        
        self.labeled_array, self.num_features = label(self.binary_ero_dil)
        
    def extract_cells(self):
        self.regions = measure.regionprops(self.labeled_array)    
        # print(len(regions))

        self.cell_list = []
        self.center_list = []
        self.Patches_Cu = []
        self.Patches_Zn = []
        self.Patches_Ca = []
        self.Patches_K = []
        self.Patches_P = []
        self.Patches_S = []
        self.Patches_Fe = []
        self.Patches_Ni = []
        self.Patches_TFY= []
        self.binary_img=[]
        self.region_vals=[]
        self.features_list = []
        self.center_coords=[]
        self.XRF_track_files=[]
        
        #motor coordinate stuff
        
        self.x_res_list= []
        self.y_res_list= []
        self.avg_res_list= []
        self.x_origin_list= []
        self.y_origin_list= []
        
        self.x_motor_center_list= []
        self.y_motor_center_list= []


        for idx in range(len(self.regions)):
            
            #append motor coordinate stuff for each region so they can be retrived later via pandas
            
            self.x_res_list.append(self.x_res)
            self.y_res_list.append(self.y_res) 
            self.avg_res_list.append(self.avg_res)
            self.x_origin_list.append(self.x_origin)
            self.y_origin_list.append(self.y_origin)
            
            #cell extraction begins here
            
            self.cell_val_bin=self.regions[idx].image
            
            self.center_coords.append(self.regions[idx].centroid)
            self.XRF_track_files.append(self.xrf_filename)

            self.region_vals.append(self.cell_val_bin)
            
            p_a = np.abs(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2))
            p_b = np.abs(math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2))
            p_c = np.abs(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))
            p_d = np.abs(math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))
            
            #debug for padding width values

#             print('p_a, p_b,p_c,p_d:',[p_a,p_b,p_c,p_d])
            
            self.padded_cell = np.pad(self.cell_val_bin, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            
            
            
            
#             self.padded_cell = np.pad(self.cell_val_bin, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
            
            self.cell_list.append(self.padded_cell)
            self.center_list.append([math.floor(self.regions[idx].centroid[0]), math.floor(self.regions[idx].centroid[1])])
            #center list is given in (y,x)
            #calculate motor centers here
            
            
            
            #motor coordinates calculated using only x_res
            self.x_motor_center_list.append(self.x_origin_list[idx] + self.x_res_list[idx]*self.center_list[idx][1])
            self.y_motor_center_list.append(self.y_origin_list[idx] + self.x_res_list[idx]*self.center_list[idx][0])
            
            
        #     regions[idx].bbox

            self.cell_Cu = self.d_Cu[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_Zn = self.d_Zn[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_Ca = self.d_Ca[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_K = self.d_K[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_P = self.d_P[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_S = self.d_S[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_Fe = self.d_Fe[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_Ni = self.d_Ni[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            self.cell_TFY = self.d_TFY[self.regions[idx].bbox[0]:self.regions[idx].bbox[2],self.regions[idx].bbox[1]:self.regions[idx].bbox[3]]
            
            self.padded_bin = np.pad(self.cell_val_bin, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            
            self.padded_Cu = np.pad(self.cell_Cu, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            self.padded_Zn = np.pad(self.cell_Zn, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            self.padded_Ca = np.pad(self.cell_Ca, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            
            self.padded_K = np.pad(self.cell_K, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            self.padded_P = np.pad(self.cell_P, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            self.padded_S = np.pad(self.cell_S, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            
            self.padded_Fe = np.pad(self.cell_Fe, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            self.padded_Ni = np.pad(self.cell_Ni, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))
            self.padded_TFY = np.pad(self.cell_TFY, ((p_a,p_b),(p_c,p_d)), mode='constant', constant_values=(0))


            
            
            
#             self.padded_bin = np.pad(self.cell_val_bin, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_Cu = np.pad(self.cell_Cu, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_Zn = np.pad(self.cell_Zn, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_Ca = np.pad(self.cell_Ca, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_K = np.pad(self.cell_K, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_P = np.pad(self.cell_P, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_S = np.pad(self.cell_S, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_Fe = np.pad(self.cell_Fe, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_Ni = np.pad(self.cell_Ni, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))
            
#             self.padded_TFY = np.pad(self.cell_TFY, ((math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[0])/2)),(math.floor((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2),math.ceil((self.BASE_PATCH_WIDTH-self.cell_val_bin.shape[1])/2))), mode='constant', constant_values=(0))

            self.binary_img.append(self.padded_bin)
            self.Patches_Cu.append(self.padded_Cu)
            self.Patches_Zn.append(self.padded_Zn)
            self.Patches_Ca.append(self.padded_Ca)
            self.Patches_K.append(self.padded_K)
            self.Patches_P.append(self.padded_P)
            self.Patches_S.append(self.padded_S)
            self.Patches_Fe.append(self.padded_Fe)
            self.Patches_Ni.append(self.padded_Ni)
            self.Patches_TFY.append(self.padded_TFY)


            # define feature vector using averages
        #     x = np.asarray([regions[idx].area, 
        #      regions[idx].eccentricity, 
        #      regions[idx].equivalent_diameter, 
        #      regions[idx].major_axis_length,
        #      regions[idx].minor_axis_length,
        #      regions[idx].perimeter,
        #      np.average(Patches_K[idx]),
        #      np.average(Patches_K[idx])/np.average(Patches_P[idx]),
        #      np.average(Patches_Ni[idx]),
        #      np.average(Patches_Ni[idx])/np.average(Patches_P[idx]),
        #     np.average(Patches_Ni[idx])/np.average(Patches_K[idx]),
        #     np.average(Patches_Cu[idx])/np.average(Patches_K[idx]),
        #     ])

            # define feature vector using averages
#             self.avg_res
    # add hand designed functions here
    def define_features(self,mode='max'):
        
        
        '''
        Defines handmade features.
        mode:, max use maximum values
        average:, use average values
        '''
        for idx in range(len(self.regions)):
            self.x = np.asarray([self.x_res*self.x_res*self.regions[idx].area, 
             self.regions[idx].eccentricity, 
             self.x_res*self.regions[idx].equivalent_diameter, 
             self.x_res*self.regions[idx].major_axis_length,
             self.x_res*self.regions[idx].minor_axis_length,
             self.x_res*self.regions[idx].perimeter,
             np.amax(self.Patches_K[idx]),
             np.amax(self.Patches_P[idx]),
             np.amax(self.Patches_Ca[idx]),
             np.amax(self.Patches_Zn[idx]),
            np.amax(self.Patches_Fe[idx]),
            np.amax(self.Patches_Cu[idx]),
            np.amax(self.Patches_TFY[idx]-self.Patches_K[idx]-self.Patches_P[idx]-self.Patches_Ca[idx]-self.Patches_Zn[idx]-self.Patches_Fe[idx]-self.Patches_Cu[idx]),
            np.unique(self.region_vals[idx], return_counts=True)[1][1] # returns the number of true (1's) values in the identified region
            ])

            self.features_list.append(self.x)
        self.features=np.asarray(self.features_list)








def calc_SNR(img, seg_img, labels = (0,1), mask_ratio = None):
    """
    SNR =  1     /  s*sqrt(std0^^2 + std1^^2)  
    where s = 1 / (mu1 - mu0)  
    mu1, std1 and mu0, std0 are the mean / std values for each of the segmented regions respectively (pix value = 1) and (pix value = 0).  
    seg_img is used as mask to determine stats in each region.  
    Parameters
    ----------
    img : np.array  
        raw input image (2D or 3D)  

    seg_img : np.array  
        segmentation map (2D or 3D)  

    labels : tuple  
        an ordered list of two label values in the image. The high value is interpreted as the signal and low value is the background.  

    mask_ratio : float or None
        If not None, a float in (0,1). The data are cropped such that the voxels / pixels outside the circular mask are ignored.  
    Returns
    -------
    float
        SNR of img w.r.t seg_img  
    """
    eps = 1.0e-12
    # handle circular mask  
    if mask_ratio is not None:
        crop_val = int(img.shape[-1]*0.5*(1 - mask_ratio/np.sqrt(2)))
        crop_slice = slice(crop_val, -crop_val)    

        if img.ndim == 2: # 2D image
            img = img[crop_slice, crop_slice]
            seg_img = seg_img[crop_slice, crop_slice]
        elif img.ndim == 3: # 3D image
            vcrop = int(img.shape[0]*(1-mask_ratio))
            vcrop_slice = slice(vcrop, -vcrop)
            img = img[vcrop_slice, crop_slice, crop_slice]
            seg_img = seg_img[vcrop_slice, crop_slice, crop_slice]

    pix_1 = img[seg_img == labels[1]]
    pix_0 = img[seg_img == labels[0]]

    if np.any(pix_1) and np.any(pix_0):
        mu1 = np.mean(pix_1)
        mu0 = np.mean(pix_0)
        s = abs(1/(mu1 - mu0 + eps))
        std1 = np.std(pix_1)
        std0 = np.std(pix_0)
        std = np.sqrt(0.5*(std1**2 + std0**2))
        std = s*std
        return 1/(std + eps)
    else:
        return 1/(np.std(img) + eps)
    

    
class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        
def accept(event):
    if event.key == "enter":
        print("Selected points:")
        print(selector.xys[selector.ind])
        selector.disconnect()
        ax.set_title("")
        fig.canvas.draw()
        
        
        
def XRF_PCA(features, feature_names, high_comp=2, n_components=2,annot_txt_size=10,dpi=75,marker_size=5,save_plots=True):
    
    '''
    Wrapper function around sklearn PCA
    arguments:
    
    features: array, contains feature for every extracted cell
    feature_names: string, names of the currently used features
    high_comp, int, an extra pca analysis to see how many components is required. Must be either number of features or number of extracted cells, whichever is the lowest.
    n_components:int, number of components chosen to do PCA
    annot_txt_size: size of text in 2D plots for marking scatter points with feature names
    dpi:dpi
    marker_size:For 3D plots
    returns
    
    principalComponents:, array, with the principal components, assign it to dataframe columns as PC1, PC2 and so on
    pca.components_: array, loading scores for the corresponding PCs
    
    '''

    X_standard = StandardScaler().fit_transform(features)
    # print(X_standard[0])

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_standard)

    
    print('Cells, PCs', principalComponents.shape)

    print('singular_values_:', pca.singular_values_)
    print('explained_variance:', pca.explained_variance_)
    print('components:', pca.components_)
    
    if n_components == 2:
    
        fig = plt.figure(figsize=(6,3),dpi=dpi);

        plt.scatter(pca.components_[0],pca.components_[1]) #, tick_label=PClabels
        plt.title('Loading Scores')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.axhline()
        plt.axvline()
        

    #     feature_names = ['area','eccentricity','equiv. dia.','major length','minor length','perimeter',
    #             'K','P','Ca','Zn',
    #              'Fe']

        for i, txt in enumerate(feature_names):
            plt.annotate(txt, (pca.components_[0][i], pca.components_[1][i]), rotation=60, size=annot_txt_size)

        plt.scatter(0,0)
#         plt.tight_layout()
        plt.show()
        if save_plots==True:
            plt.savefig('../figures/PCA_ls.jpg')
    
    if n_components == 3:
#         sns.set(style = "darkgrid")

        fig = px.scatter_3d(x=pca.components_[0], y=pca.components_[1], z=pca.components_[2], text=feature_names)
        fig.update_traces(marker_size = marker_size)
        fig.update_layout(scene = dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'),
                    width=700)
#         fig.tight_layout()
        fig.show()
        if save_plots==True:
            fig.write_image('../figures/PCA_ls_3D.jpg')
        
        
    
    #scree plot

    high_pca = PCA(n_components=high_comp)
    high_pca.fit_transform(X_standard)

    #calculate percentage of variation in each principal components
    per_var=np.round(high_pca.explained_variance_ratio_*100, decimals=1)
    PClabels =['PC' + str(x) for x in range(1,len(per_var)+1)]

    fig = plt.figure(figsize=(6,3),dpi=dpi);
    plt.bar(x=range(1, len(per_var)+1),height=per_var) #, tick_label=PClabels
    plt.title('Scree Plot')
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.tight_layout()
    plt.show()
    
    if save_plots==True:
        fig.savefig('../figures/PCA_variance.jpg')
        
    #zoom in on the important PCs
    fig = plt.figure(figsize=(6,3),dpi=dpi);
    plt.bar(x=range(1, len(per_var)+1),height=per_var) #, tick_label=PClabels
    plt.title('Scree Plot (Significnt PCs)')
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.xlim(0,75)
    plt.tight_layout()
    plt.show()
    if save_plots==True:
        fig.savefig('../figures/High_PCA_variance.jpg')
    
    if n_components == 2:
        plt.figure(dpi=dpi)
        plt.scatter(principalComponents[:,0],principalComponents[:,1], s=10)
        plt.title('PCA-space untagged')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        if save_plots==True:
            plt.savefig('../figures/PCA_space.jpg')
        
    if n_components == 3:
            fig = px.scatter_3d(x=principalComponents[:,0], y=principalComponents[:,1], z=principalComponents[:,2])
            fig.update_traces(marker_size = marker_size)
            fig.update_layout(scene = dict(
                        xaxis_title='PC1',
                        yaxis_title='PC2',
                        zaxis_title='PC3'),
                        width=700)
#             fig.tight_layout()
            fig.show()
            if save_plots==True:
                fig.write_image('../figures/PCA_space_3D.jpg')
    
    
    return principalComponents, pca.components_

def view_PC_feature_relation(dataframe, PC_names, feature_list, hue,save_plots=True):
    '''
    argument:
    
    dataframe: pandas dataframe, consists of PC1, PC2 ... and so on, and the features used to construct the dataframe.
    PC_names: list, list of the columns containing PCs
    feature_list: list, list of the columns we wish to plot. It is recommended to pass two original features at a time for better results.
    hue:str, column name of the pandas dataframe for tagging scatter points in the pairplot
    
    returns:
    seaborn pairplots of showing correlations amongst PC and the original features.
    '''
    

    sns.pairplot(
        dataframe, hue=hue,
#         vars=PC_names+feature_list,
        x_vars=feature_list,
        y_vars=PC_names, height=4,corner=False
    )
    plt.tight_layout()
    if save_plots==True:
                plt.savefig('../figures/PCA_features_rel.jpg')
            
            
def remove_artifacts(principalDf, remove_count = 8):
    '''
    remove artifact cells based on pixel criteria
    
    arguments:
    principalDf: pandas dataframe, contains all the defined features for each extracted regions.
    remove_count: int, pixel threshold below which regions are considered artifacts.
    
    returns:
    secondaryDf: pandas dataframe, contains only the regions deemed actual cells and associated feautres 
    
    '''
    
    secondaryDf=principalDf

    # remove additional artifacts that do not contain remove_count pixels
    secondaryDf=secondaryDf[secondaryDf['Pixel_count'] >remove_count]

    #keep track of the original indices after removing artifacts
    secondaryDf['original index'] = secondaryDf.index.to_numpy()
    secondaryDf.reset_index(drop=True, inplace=True)
    
    return secondaryDf
    

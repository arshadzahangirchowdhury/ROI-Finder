#imports
from roifinder.misc.patches2d import Patches as Patches2D

import cv2
import os, h5py, collections, sys
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import clear_output
from ipywidgets import interactive


from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage import measurements,morphology 

from scipy.ndimage import binary_erosion, binary_dilation

from skimage import data
from skimage.filters import threshold_otsu

import math
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans
import skimage.filters
import tifffile as tiff




class CenterSampling():
    """Gets center of the bacterial cell from the xrf image."""
    
    def __init__(self, BASE_PATCH_WIDTH):
        """
        
        """
        
        self.BASE_PATCH_WIDTH=BASE_PATCH_WIDTH
        
        return 

    def get_cell_centers(self, img, binary_ero_dil):
        '''
        Parameters
        ----------
        img : (N, M) array,
            input non-binary image.
        binary_ero_dil : (N, M) array,
            Processed binary image.
        BASE_PATCH_WIDTH : int,
        Length or side of a square patch.

        Returns
        -------
        y_corner_cells, x_corner_cells, y_centers_cells, x_centers_cells : topleft corner and center points' coordinates
        
        '''
        
        HALF_WIDTH=self.BASE_PATCH_WIDTH/2
        #No. of components are calculated from eroded-dilated image and are arranged in label array
        labeled_array, num_features = label(binary_ero_dil)


        #Check if the components (Ecoli cell) are tagged
    #     print(np.unique(labeled_array))

        #Print the number of (approximate) cell which are there 
    #     print('num_of_cells = ', num_features)

        #Initialize store array for indices with shape
        indices_for_cells=np.empty((num_features), dtype=object)
        x_centers_cells=np.empty((num_features), dtype=object)
        y_centers_cells=np.empty((num_features), dtype=object)

        x_centers_cells_list=[]
        y_centers_cells_list=[]

        for j in range(0, num_features):
        #     print(j)
            #indices for cells array stores all the indices identified from the labeling

            np.sum(labeled_array == j + 1)


            indices_for_cells[j]=np.where(labeled_array == j + 1)
            #note centers have to be flipped for convention
            x_centers_cells[j]=math.ceil((indices_for_cells[j][1].mean()))
            y_centers_cells[j]=math.ceil(indices_for_cells[j][0].mean())
            
        #Calculate corner points
        y_corner_cells, x_corner_cells = (y_centers_cells - HALF_WIDTH), (x_centers_cells-HALF_WIDTH)

        #Adjust corner points
        y_corner_cells[y_corner_cells<0]=0
        x_corner_cells[x_corner_cells<0]=0
        x_corner_cells[x_corner_cells+self.BASE_PATCH_WIDTH> binary_ero_dil.shape[1]]=img.shape[1]-self.BASE_PATCH_WIDTH
        y_corner_cells[(y_corner_cells+self.BASE_PATCH_WIDTH> binary_ero_dil.shape[0])]=img.shape[0]-self.BASE_PATCH_WIDTH



        return y_corner_cells, x_corner_cells, y_centers_cells, x_centers_cells
    

    
def check_cell(self, img, binary_ero_dil):
        '''
        Parameters
        ----------
        img : (N, M) array,
            input non-binary image.
        binary_ero_dil : (N, M) array,
            Processed binary image.
        BASE_PATCH_WIDTH : int,
        Length or side of a square patch.

        Returns
        -------
        y_corner_cells, x_corner_cells, y_centers_cells, x_centers_cells : topleft corner and center points' coordinates
        
        '''
        
        HALF_WIDTH=self.BASE_PATCH_WIDTH/2
        #No. of components are calculated from eroded-dilated image and are arranged in label array
        labeled_array, num_features = label(binary_ero_dil)


        #Check if the components (Ecoli cell) are tagged
    #     print(np.unique(labeled_array))

        #Print the number of (approximate) cell which are there 
    #     print('num_of_cells = ', num_features)

        #Initialize store array for indices with shape
        indices_for_cells=np.empty((num_features), dtype=object)
        x_centers_cells=np.empty((num_features), dtype=object)
        y_centers_cells=np.empty((num_features), dtype=object)

        x_centers_cells_list=[]
        y_centers_cells_list=[]

        for j in range(0, num_features):
        #     print(j)
            #indices for cells array stores all the indices identified from the labeling

            np.sum(labeled_array == j + 1)


            indices_for_cells[j]=np.where(labeled_array == j + 1)
            #note centers have to be flipped for convention
            x_centers_cells[j]=math.ceil((indices_for_cells[j][1].mean()))
            y_centers_cells[j]=math.ceil(indices_for_cells[j][0].mean())
            
        #Calculate corner points
        y_corner_cells, x_corner_cells = (y_centers_cells - HALF_WIDTH), (x_centers_cells-HALF_WIDTH)

        #Adjust corner points
        y_corner_cells[y_corner_cells<0]=0
        x_corner_cells[x_corner_cells<0]=0
        x_corner_cells[x_corner_cells+self.BASE_PATCH_WIDTH> binary_ero_dil.shape[1]]=img.shape[1]-self.BASE_PATCH_WIDTH
        y_corner_cells[(y_corner_cells+self.BASE_PATCH_WIDTH> binary_ero_dil.shape[0])]=img.shape[0]-self.BASE_PATCH_WIDTH



        return y_corner_cells, x_corner_cells, y_centers_cells, x_centers_cells
        

class ClusterAnalysis():
    # K-mean cluster analysis
    def kmean_analysis(n_clusters, data, random_state, sigma = 1, cval = None,
                       plotoption = None, savefig = None, figsize = (10, 10), fname = None):

        data[np.isnan(data)] = 1e-5
        data[np.isinf(data)] = 1e-5

        if sigma is not None:
            data_blur = skimage.filters.gaussian(data, sigma = sigma)
        else:
            data_blur = data

        km = KMeans(n_clusters = n_clusters,random_state=random_state)
        km.fit(data_blur.reshape(-1,1))

        km_label = np.reshape(km.labels_, data.shape)

        # sort label based on center
        srtIndex = np.argsort(km.cluster_centers_[:,0])
        for i, s in enumerate(srtIndex):
            km_label[km_label == s] = -(i+1)
        km_label = np.multiply(-1,km_label)-1
        km_bool = km_label.copy()
        km_bool[km_bool > 1] = 1

        fig = None

        if plotoption:
            fig, ax = plt.subplots(1,4,figsize=figsize)
            a = ax[0].imshow(data, cmap = plt.cm.get_cmap('Greys_r'))
            if cval == None:
                cval = a.get_clim()
            else:
                a.set_clim(cval)
            c = ax[1].imshow(data_blur, cmap = plt.cm.get_cmap('inferno'))
            k = ax[2].imshow(km_label, vmin = 0, vmax = n_clusters-1)
            b = ax[3].imshow(np.multiply(data,km_bool), cmap = plt.cm.get_cmap('Greys_r'))
            b.set_clim(cval)
            c.set_clim(cval)
            fig.colorbar(a, ax=ax[0], orientation='horizontal', shrink = 0.8)
            fig.colorbar(c, ax = ax[1], orientation='horizontal', shrink = 0.8)
            fig.colorbar(k, ax = ax[2], orientation='horizontal', shrink = 0.8)
            fig.colorbar(b, ax = ax[3], orientation = 'horizontal', shrink = 0.8)


            map_label = ['data','blur', 'blur-kmean', 'data * kmean']
            for ax_, l in zip(ax, map_label):
                ax_.axis('off')
                ax_.axis('equal')
                ax_.set_title(l)
                # ax_.axis('scaled')
    #            ax_.text(2, 7, l, color = 'w')
            plt.tight_layout()
            plt.show()

            if (savefig == 1) & (fname is not None):
                fig.savefig(fname, dpi = 300)

        return km, km_label, km_bool, fig




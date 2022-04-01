#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for processing xrf data and class implementation of the autoencoder models.

"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#imports
import cv2, os, h5py, collections, sys, math
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import clear_output
from ipywidgets import interactive
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure,measurements,morphology 
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import data
from skimage.filters import threshold_otsu
import seaborn as sns
import pdb 


sys.path.append('/data02/AZC/AI-XRF/roi_finder')


from tools.misc.Utils import CenterSampling, ClusterAnalysis
from tools.misc.patches2d import Patches as Patches2D


# from Utils import CenterSampling, ClusterAnalysis
# from patches2d import Patches2D






#Function names should not start with capital

def _processor(data_format, files_in_location,tif_train,BASE_PATCH_WIDTH):
    '''
    Loads a image, converts it to binary , finds the features in the binary image, extract subimages and returns them.

    Args:
    -----
    data_format: The desired output format of the subimages
      - 'grayscale': returns the grayscale values for the subimages
      - 'binary': returns the binary values for the subimages
    files_in_location: list of tif files
    tif_train: class-specific list of files
    BASE_PATCH_WIDTH: size of subimage
    
    returns:
    --------
    Returns subimages in either binary or grayscale format
      
      
    '''
    
    Stored_Patches=np.empty([0,BASE_PATCH_WIDTH*BASE_PATCH_WIDTH])
    for idx, f in enumerate(files_in_location):
          
        if '.tif' in f:
            filename=f


            data = tiff.imread(os.path.join(tif_train, f))

            data_original=data

            data = ndimage.median_filter(data, size=4)


            thresh = threshold_otsu(data)
            binary = data > thresh
            #invert image

            binary=binary^1
            #Also do bit flip


            binary_eroded=ndimage.binary_erosion(binary).astype(binary.dtype)
#             binary_dilated=ndimage.binary_dilation(binary).astype(binary.dtype)
            binary_ero_dil=ndimage.binary_dilation(binary_eroded).astype(binary_eroded.dtype)



            labeled_array, num_features = label(binary_ero_dil)

            y_corner_cells, x_corner_cells, y_centers_cells, x_centers_cells = CenterSampling(BASE_PATCH_WIDTH).get_cell_centers(data, 
                                              binary_ero_dil

                                              )

            mini_patch_size=(BASE_PATCH_WIDTH,BASE_PATCH_WIDTH)
            #either this or the transpose of this

            widths=np.array([mini_patch_size]*len(x_centers_cells))

            points=np.array([y_corner_cells,x_corner_cells]).T.astype(int)

            p = Patches2D(data.shape, \
                          points=points, \
                          widths=widths, \
                          initialize_by = "data", 
                          )
            
            
            if data_format=='grayscale':
                Patches = p.extract(data_original, mini_patch_size) #This gives gray scale image.
            
            elif data_format=='binary':
                Patches = p.extract(binary_ero_dil, mini_patch_size) #This gives gray scale image.            
    
    
        Patches=Patches.reshape(Patches.shape[0],BASE_PATCH_WIDTH*BASE_PATCH_WIDTH)

            
        
        Stored_Patches=np.concatenate((Stored_Patches,Patches),axis=0)
    
    return Stored_Patches.reshape(Stored_Patches.shape[0], BASE_PATCH_WIDTH, BASE_PATCH_WIDTH)




def load_LIVECell(data_format,BASE_PATCH_WIDTH=28):
    '''
    Loads LIVECell data.

    Args:
    -----
    data_format: The desired output format of the subimages
      - 'grayscale': returns the grayscale values for the subimages
      - 'binary': returns the binary values for the subimages
    
    BASE_PATCH_WIDTH: size of subimage
    
    returns:
    --------
    Returns subimages in either binary or grayscale format from the LIVECell database.
      
      
    '''
    
    ####
    
    #Function arguments
    tiffdir = 'output/'
    normOnly = True

    ####

    cwd = os.getcwd()

    dir_names=cwd.split(os.sep)


    #just combining the directory names to create a path
    tiff_train=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images')

    tiff_test=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images')

    tif_train_A172=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'A172')
    
    
    tif_train_A172=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'A172')
    tif_train_BT474=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'BT474')
    tif_train_BV2=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'BV2')
    tif_train_Huh7=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'Huh7')
    tif_train_MCF7=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'MCF7')
    tif_train_SHSY5Y=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'SHSY5Y')
    tif_train_SKOV3=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'SKOV3')
    tif_train_SkBr3=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_train_val_images', 'SkBr3')


    files_train_A172 = os.listdir(tif_train_A172)
    files_train_BT474 = os.listdir(tif_train_BT474)
    files_train_BV2 = os.listdir(tif_train_BV2)
    files_train_Huh7 = os.listdir(tif_train_Huh7)
    files_train_MCF7 = os.listdir(tif_train_MCF7)
    files_train_SHSY5Y = os.listdir(tif_train_SHSY5Y)
    files_train_SKOV3 = os.listdir(tif_train_SKOV3)
    files_train_SkBr3 = os.listdir(tif_train_SkBr3)
    
    
    tif_test_A172=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'A172')
    tif_test_BT474=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'BT474')
    tif_test_BV2=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'BV2')
    tif_test_Huh7=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'Huh7')
    tif_test_MCF7=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'MCF7')
    tif_test_SHSY5Y=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'SHSY5Y')
    tif_test_SKOV3=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'SKOV3')
    tif_test_SkBr3=os.path.join('/',dir_names[0],dir_names[1],dir_names[2],'LIVECell','livecell_test_images', 'SkBr3')



    files_test_A172 = os.listdir(tif_test_A172)
    files_test_BT474 = os.listdir(tif_test_BT474)
    files_test_BV2 = os.listdir(tif_test_BV2)
    files_test_Huh7 = os.listdir(tif_test_Huh7)
    files_test_MCF7 = os.listdir(tif_test_MCF7)
    files_test_SHSY5Y = os.listdir(tif_test_SHSY5Y)
    files_test_SKOV3 = os.listdir(tif_test_SKOV3)
    files_test_SkBr3 = os.listdir(tif_test_SkBr3)

    
    
    A172_train_Patches=_processor(data_format,files_train_A172,tif_train_A172,BASE_PATCH_WIDTH)
    BT474_train_Patches=_processor(data_format,files_train_BT474,tif_train_BT474,BASE_PATCH_WIDTH)
    BV2_train_Patches=_processor(data_format,files_train_BV2,tif_train_BV2,BASE_PATCH_WIDTH)
    Huh7_train_Patches=_processor(data_format,files_train_Huh7,tif_train_Huh7,BASE_PATCH_WIDTH)
    MCF7_train_Patches=_processor(data_format,files_train_MCF7,tif_train_MCF7,BASE_PATCH_WIDTH)
    SHSY5Y_train_Patches=_processor(data_format,files_train_SHSY5Y,tif_train_SHSY5Y,BASE_PATCH_WIDTH)
    SKOV3_train_Patches=_processor(data_format,files_train_SKOV3,tif_train_SKOV3,BASE_PATCH_WIDTH)
    SkBr3_train_Patches=_processor(data_format,files_train_SkBr3,tif_train_SkBr3,BASE_PATCH_WIDTH)
    
    
    
    y_A172_train=np.empty([A172_train_Patches.shape[0] ] )
    y_BT474_train=np.empty([BT474_train_Patches.shape[0] ] )
    y_BV2_train=np.empty([BV2_train_Patches.shape[0] ] )
    y_Huh7_train=np.empty([Huh7_train_Patches.shape[0] ] )
    y_MCF7_train=np.empty([MCF7_train_Patches.shape[0] ] )
    y_SHSY5Y_train=np.empty([SHSY5Y_train_Patches.shape[0] ] )
    y_SKOV3_train=np.empty([SKOV3_train_Patches.shape[0] ] )
    y_SkBr3_train=np.empty([SkBr3_train_Patches.shape[0] ] )


    for i in range(A172_train_Patches.shape[0]):
        y_A172_train[i]=0

    for i in range(BT474_train_Patches.shape[0]):
        y_BT474_train[i]=1

    for i in range(BV2_train_Patches.shape[0]):
        y_BV2_train[i]=2

    for i in range(Huh7_train_Patches.shape[0]):
        y_Huh7_train[i]=3

    for i in range(MCF7_train_Patches.shape[0]):
        y_MCF7_train[i]=4

    for i in range(SHSY5Y_train_Patches.shape[0]):
        y_SHSY5Y_train[i]=5

    for i in range(SKOV3_train_Patches.shape[0]):
        y_SKOV3_train[i]=6

    for i in range(SkBr3_train_Patches.shape[0]):
        y_SkBr3_train[i]=7

    


    A172_test_Patches=_processor(data_format,files_test_A172,tif_test_A172,BASE_PATCH_WIDTH)
    BT474_test_Patches=_processor(data_format,files_test_BT474,tif_test_BT474,BASE_PATCH_WIDTH)
    BV2_test_Patches=_processor(data_format,files_test_BV2,tif_test_BV2,BASE_PATCH_WIDTH)
    Huh7_test_Patches=_processor(data_format,files_test_Huh7,tif_test_Huh7,BASE_PATCH_WIDTH)
    MCF7_test_Patches=_processor(data_format,files_test_MCF7,tif_test_MCF7,BASE_PATCH_WIDTH)
    SHSY5Y_test_Patches=_processor(data_format,files_test_SHSY5Y,tif_test_SHSY5Y,BASE_PATCH_WIDTH)
    SKOV3_test_Patches=_processor(data_format,files_test_SKOV3,tif_test_SKOV3,BASE_PATCH_WIDTH)
    SkBr3_test_Patches=_processor(data_format,files_test_SkBr3,tif_test_SkBr3,BASE_PATCH_WIDTH)
    
    y_A172_test=np.empty([A172_test_Patches.shape[0] ] )
    y_BT474_test=np.empty([BT474_test_Patches.shape[0] ] )
    y_BV2_test=np.empty([BV2_test_Patches.shape[0] ] )
    y_Huh7_test=np.empty([Huh7_test_Patches.shape[0] ] )
    y_MCF7_test=np.empty([MCF7_test_Patches.shape[0] ] )
    y_SHSY5Y_test=np.empty([SHSY5Y_test_Patches.shape[0] ] )
    y_SKOV3_test=np.empty([SKOV3_test_Patches.shape[0] ] )
    y_SkBr3_test=np.empty([SkBr3_test_Patches.shape[0] ] )


    for i in range(A172_test_Patches.shape[0]):
        y_A172_test[i]=0

    for i in range(BT474_test_Patches.shape[0]):
        y_BT474_test[i]=1

    for i in range(BV2_test_Patches.shape[0]):
        y_BV2_test[i]=2

    for i in range(Huh7_test_Patches.shape[0]):
        y_Huh7_test[i]=3

    for i in range(MCF7_test_Patches.shape[0]):
        y_MCF7_test[i]=4

    for i in range(SHSY5Y_test_Patches.shape[0]):
        y_SHSY5Y_test[i]=5

    for i in range(SKOV3_test_Patches.shape[0]):
        y_SKOV3_test[i]=6

    for i in range(SkBr3_test_Patches.shape[0]):
        y_SkBr3_test[i]=7

    y_test = np.concatenate((y_A172_test,y_BT474_test,y_BV2_test,y_Huh7_test,y_MCF7_test,y_SHSY5Y_test,y_SKOV3_test,y_SkBr3_test), axis =0)
    
    y_train = np.concatenate((y_A172_train,y_BT474_train,y_BV2_train,y_Huh7_train,y_MCF7_train,y_SHSY5Y_train,y_SKOV3_train,y_SkBr3_train), axis =0)
    
    x_train = np.concatenate((A172_train_Patches,BT474_train_Patches,BV2_train_Patches,Huh7_train_Patches,MCF7_train_Patches,SHSY5Y_train_Patches,SKOV3_train_Patches,SkBr3_train_Patches), axis =0)

    x_test = np.concatenate((A172_test_Patches,BT474_test_Patches,BV2_test_Patches,Huh7_test_Patches,MCF7_test_Patches,SHSY5Y_test_Patches,SKOV3_test_Patches,SkBr3_test_Patches), axis = 0)
    
    
    return x_train, y_train, x_test, y_test

def noise_addition(data_format,noise_amplitude,x_train,x_test):
    '''
    Adds random noise to training and testing datasets.

    Args:
    -----
    data_format: The desired output format of the subimages in x_train and x_test
      - 'grayscale': returns the grayscale values for the subimages
      - 'binary': returns the binary values for the subimages
    noise_amplitude: amplitude of the noise
    x_train: training images
    x_test: testing images

    
    returns:
    --------
    Returns subimages in either binary or grayscale format with added random noise
      
      
    '''
    
    x_train_noisy = x_train + noise_amplitude * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_amplitude * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    
    
    if data_format=='grayscale':
        x_train_noisy = np.clip(x_train_noisy, 0., 255.)
        x_test_noisy = np.clip(x_test_noisy, 0., 255.)
            
    elif data_format=='binary':
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    
    return x_train_noisy,x_test_noisy

def single_noise_addition(data_format,noise_amplitude,x_train):
    '''
    Adds random noise to training and testing datasets.

    Args:
    -----
    data_format: The desired output format of the subimages in x_train
      - 'grayscale': returns the grayscale values for the subimages
      - 'binary': returns the binary values for the subimages
    noise_amplitude: amplitude of the noise
    x_train: training images
    
    
    returns:
    --------
    Returns subimages in either binary or grayscale format with added random noise
      
      
    '''
    
    x_train_noisy = x_train + noise_amplitude * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    
    
    if data_format=='grayscale':
        x_train_noisy = np.clip(x_train_noisy, 0., 255.)
        
            
    elif data_format=='binary':
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        
    
    
    return x_train_noisy



def combine_data(x_train,x_test,y_train, y_test):  
    '''
    Combines train and test datasets.

    Args:
    -----

    x_train: training images
    x_test: testing images
    y_train: training labels
    y_test: testing labels


    
    returns:
    --------
    Returns combined data and labels
      
    '''
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    combined_data = np.concatenate([x_train, x_test], axis=0)
    combined_label = np.concatenate([y_train, y_test], axis=0)



    print('(Sub) Images shape :',combined_data.shape)
    print('Annotations shape  :',combined_label.shape)
    
    return combined_data,combined_label


def plot_3Dprojection(df, figw = 8):
    '''
    Plots 3D scatterplot of the dataframe
    Args:
    -----
    df: Pandas dataframe with 3 columns and a fourth column labeling the data.
      
    fig_w: width of the 3D scatterplot.
    
    returns:
    --------
    Returns 3D scatterplot labeled with class types
    '''
      
    
    fig = plt.figure(figsize = (figw,figw))
    ax = fig.add_subplot(111, projection = '3d')
    markers= ('x', 'o', '>', '<', 's', 'v', 'H', 'D', '3', '1', '2')
    # colors = ('g', 'b', 'gold', 'yellow', 'tan', 'cyan', 'magenta', 'black', 'orange', 'darkgreen')
    colors = ('g', 'b', 'gold', 'tan', 'magenta', 'black', 'orange', 'darkgreen')

    data_labels = list(df["label"].unique())
    for idx, lab in enumerate(data_labels):

        ax.scatter(df[df["label"] == lab]["$z_1$"], \
                   df[df["label"] == lab]["$z_0$"], \
                   df[df["label"] == lab]["$z_2$"], \
                   marker = markers[0], c = colors[idx], s = 10, label = lab)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    ax.legend(bbox_to_anchor=(0., 1.0, 1., .1), ncol=2, loc=3, fancybox=False, framealpha=0.5, fontsize=10)
    plt.subplots_adjust(top = 0.8)    
    plt.tight_layout()

    return

def check_VAE_output(combined_data,image_no,BASE_PATCH_WIDTH,encoder,decoder):
    '''
    Reads an image from the combinded dataset and compares against VAE prediction
    
    Args:
    -----
    combined_data: The image dataset
      
    image_no: index of the input image
    
    BASE_PATCH_WIDTH: index of the input image
    
    encoder: encoder model
    
    decoder: decoder model
    
    returns:
    --------
    Returns the plots of the input image and the image returned by the decoder.
    
    '''
    
    plt.figure(figsize=(6,6))
    print("image no: ", image_no)
    plt.subplot(221)
    plt.title('Input image')
    plt.imshow(combined_data[image_no]);
    plt.tight_layout()

    plt.subplot(222)
    plt.title('Decoder output')
    outputs = decoder.predict(encoder.predict(combined_data[image_no].reshape(1,BASE_PATCH_WIDTH,BASE_PATCH_WIDTH,1))[2])
    outputs=np.round(outputs)
    plt.imshow(outputs.reshape(BASE_PATCH_WIDTH,BASE_PATCH_WIDTH,1));
    plt.tight_layout()
    
    
    
def view_latent_variable_clusters(combined_data,combined_label,cell_types,encoder, pairplot_height=3, PCA_flag='yes',PCA_figwidth=16): 
    
    '''
    Reads an image from the combinded dataset and compares against VAE prediction.
    
    Args:
    -----
    combined_data: The image dataset.
      
    combined_label: The image dataset labels or annotations.
    
    cell_types: Number of classes or types of cells present in the image dataset.
    
    encoder: encoder model.
    
    pairplot_height: height of the pairplot figure.
    
    PCA_flag: Flag variable to turn on or off the PCA analysis.
    
    PCA_figwidth: width of the PCA scatterplot figure.
    
    returns:
    --------
    Returns the plots of the input image and the image returned by the decoder.
    
    '''
    
    
    z = encoder.predict(combined_data)[2]
    df_z=pd.DataFrame(z)
    df_z['CellType']=combined_label.astype(int)
    # df_z
    custom_palette = sns.color_palette("Paired", cell_types)
    # ax=sns.pairplot(df_z, hue='CellType',palette=custom_palette, corner=True, height=pairplot_height)
    ax=sns.pairplot(df_z, hue='CellType', corner=True, height=pairplot_height)
    
    if PCA_flag=='yes':
        from sklearn.decomposition import PCA

        if z.shape[0]>3:
            
            pca = PCA(n_components=3)
            pca.fit(z)

            print('PCA explained_variance_ratio :',pca.explained_variance_ratio_)

            print('PCA singular_values :',pca.singular_values_)

            PCA_Outputs=pca.transform(z)

            df=pd.DataFrame(columns=['$z_0$', '$z_1$','$z_2$', 'label'], data=np.concatenate([PCA_Outputs, combined_label.reshape(-1,1)], axis=1 ))

            plot_3Dprojection(df, figw = PCA_figwidth)

            sns.pairplot(df, hue="label", corner=True, height=pairplot_height)
            
            sns.pairplot(df.loc[df['label'] == 0], hue="label",palette=['navy'] ,corner=True, height=pairplot_height*0.5)
            sns.pairplot(df.loc[df['label'] == 1], hue="label",palette=['darkorange'] ,corner=True, height=pairplot_height*0.5)



            
            
            

        
        
        




def model_history(vae, regularization_type):
    
    '''
    Plots the model losses vs epoch.
    
    Args:
    -----
    vae: The autoencoder model.
      
    regularization_type: Type of regularization of model loss.
    -'kl': Kullback-Leibler divergence loss
    -'L1': L1 loss.
    
    
    
    returns:
    --------
    Returns the plots of the total loss, reconstruction loss and the regularization loss.
    
    '''
    
    
    f = plt.figure(figsize=(10,3))
    ax = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)



    ax.plot(vae.history['loss'])
    ax.set_title('total loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['loss'], loc='upper right')

    ax2.plot(vae.history['reconstruction_loss'])
    ax2.set_title('recon')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['recon'], loc='upper right')

    if regularization_type == 'L1':
        
        ax3.plot(vae.history['regularization_loss'])
        ax3.set_title('L1')
        ax3.set_ylabel('loss')
        ax3.set_xlabel('epoch')
        ax3.legend(['L1'], loc='upper right')
        
    elif regularization_type == 'kl':
        
        ax3.plot(vae.history['regularization_loss'])
        ax3.set_title('kl')
        ax3.set_ylabel('loss')
        ax3.set_xlabel('epoch')
        ax3.legend(['kl'], loc='upper right')
        
    plt.tight_layout()
    plt.show()

    
    
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a subimage."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def autoencoder(latent_dim,num_channels,BASE_PATCH_WIDTH,summary='yes'):
    
    '''
    Defines the autoencoder model.
    
    Args:
    -----
    latent_dim: Dimension of the latent vector.
    
    num_channels: number of channels
    
    BASE_PATCH_WIDTH: size of the subimage.
    
    summary: (optional) print model summary.
      
    
    returns:
    --------
    Returns encoder and the decoder models.
    
    '''
    
    
    #Architechture

    encoder_inputs = keras.Input(shape=(BASE_PATCH_WIDTH, BASE_PATCH_WIDTH, num_channels))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    if summary=='yes':
        encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    if summary=='yes':
        decoder.summary()
    
    return encoder,decoder

def autoencoder_3D(latent_dim,num_channels,BASE_PATCH_WIDTH,summary='yes'):
    
    '''
    Defines the 3D autoencoder model.
    
    Args:
    -----
    latent_dim: Dimension of the latent vector.
    
    num_channels: number of channels
    
    BASE_PATCH_WIDTH: size of the subimage.
    
    summary: (optional) print model summary.
      
    
    returns:
    --------
    Returns encoder and the decoder models.
    
    '''
    
    
    #Architechture

    encoder_inputs = keras.Input(shape=(BASE_PATCH_WIDTH, BASE_PATCH_WIDTH, num_channels))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    if summary=='yes':
        encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(num_channels, 3, activation="relu", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    if summary=='yes':
        decoder.summary()
    
    return encoder,decoder


def autoencoder_3D_deep(latent_dim,num_channels,BASE_PATCH_WIDTH,summary='yes'):
    
    '''
    Defines the 3D autoencoder model with more convolutional layers.
    
    Args:
    -----
    latent_dim: Dimension of the latent vector.
    
    num_channels: number of channels
    
    BASE_PATCH_WIDTH: size of the subimage.
    
    summary: (optional) print model summary.
      
    
    returns:
    --------
    Returns encoder and the decoder models.
    
    '''
    
    
    #Architechture

    encoder_inputs = keras.Input(shape=(BASE_PATCH_WIDTH, BASE_PATCH_WIDTH, num_channels))
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    if summary=='yes':
        encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#     x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(num_channels, 3, activation="relu", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    if summary=='yes':
        decoder.summary()
    
    return encoder,decoder


def check_VAE_3D_output(combined_data,image_no,BASE_PATCH_WIDTH,encoder,decoder):
    '''
    Reads an image from the combinded dataset and compares against a 3D VAE prediction
    
    Args:
    -----
    combined_data: The image dataset
      
    image_no: index of the input image
    
    BASE_PATCH_WIDTH: index of the input image
    
    encoder: encoder model
    
    decoder: decoder model
    
    returns:
    --------
    Returns the plots of the input image and the image returned by the decoder.
    
    '''
    
    plt.figure(figsize=(8,8))
    print("image no: ", image_no)
    plt.subplot(231)
    plt.title('Input image: Cu')
    plt.imshow(combined_data.T[0].T[image_no]);
    plt.subplot(232)
    plt.title('Input image: Ca')
    plt.imshow(combined_data.T[1].T[image_no]);
    plt.subplot(233)
    plt.title('Input image: K')
    plt.imshow(combined_data.T[2].T[image_no]);

    decoder_output=decoder.predict(encoder.predict(np.expand_dims(combined_data[image_no],0))[2])

    plt.subplot(234)
    plt.title('Output image: Cu')
    plt.imshow(decoder_output.T[0].T[0]);
    plt.subplot(235)
    plt.title('Output image: Ca')
    plt.imshow(decoder_output.T[1].T[0]);
    plt.subplot(236)
    plt.title('Output image: K')
    plt.imshow(decoder_output.T[2].T[0]);


    plt.tight_layout()
        

class Short_VAE(keras.Model):
    """Modifies the keras.Model to implement custom loss functions and train step
    
    
    
    Args:
    -----
    encoder: the encoder model.
    
    decoder: the decoder model.
    
    weight: stregnth of the regularization loss (L1 or KL).
    
    regularization_type:  Type of regularization of model loss.
    -'kl': Kullback-Leibler divergence loss
    -'L1': L1 loss.
    
    
    
    
    """
    def __init__(self, encoder, decoder,weight=1/250,regularization_type='kl',recon_type='bce', **kwargs):
        super(Short_VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.weight=weight
        self.regularization_type=regularization_type
        self.recon_type=recon_type
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
#         self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
#         self.beta_kl_loss_tracker = keras.metrics.Mean(name="beta_kl_loss")
        self.regularization_loss_tracker = keras.metrics.Mean(name="regularization_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
#             self.kl_loss_tracker,
#             self.beta_kl_loss_tracker,
            self.regularization_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            if self.recon_type=='mse':
                
                reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(data, reconstruction))
                
            elif self.recon_type=='bce':
                
                reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction))
                
            else:
                raise ValueError("Reconstruction loss must be either 'mse' or 'bcel' " )
            
            if self.regularization_type=='L1':
                regularization_loss=tf.reduce_mean(tf.abs(z))
                
            elif self.regularization_type=='kl':
                regularization_loss = tf.reduce_mean(keras.losses.kl_divergence(data, reconstruction))
                
            else:
                raise ValueError("Regularization loss must be either 'L1' or 'kl' " )
#             weight=1/250


            #Print the losses
#             pdb.set_trace()
            total_loss = reconstruction_loss + self.weight*regularization_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)






        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "regularization_loss": self.regularization_loss_tracker.result()
        }





class VAE(keras.Model):
    """Modifies the keras.Model to implement custom loss functions and train step
    
    
    
    Args:
    -----
    encoder: the encoder model.
    
    decoder: the decoder model.
    
    weight: stregnth of the regularization loss (L1 or KL).
    
    regularization_type:  Type of regularization of model loss.
    -'kl': Kullback-Leibler divergence loss
    -'L1': L1 loss.
    
    
    
    
    """
    def __init__(self, encoder, decoder,weight=1/250,regularization_type='kl',recon_type='bce', **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.weight=weight
        self.regularization_type=regularization_type
        self.recon_type=recon_type
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
#         self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
#         self.beta_kl_loss_tracker = keras.metrics.Mean(name="beta_kl_loss")
        self.regularization_loss_tracker = keras.metrics.Mean(name="regularization_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
#             self.kl_loss_tracker,
#             self.beta_kl_loss_tracker,
            self.regularization_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            data_x, data_y = data
            z_mean, z_log_var, z = self.encoder(data_x)
            reconstruction = self.decoder(z)
            
            if self.recon_type=='mse':
                
                reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(data_y, reconstruction))
                
            elif self.recon_type=='bce':
                
                reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data_y, reconstruction))
                
            else:
                raise ValueError("Reconstruction loss must be either 'mse' or 'bcel' " )
            
            if self.regularization_type=='L1':
                regularization_loss=tf.reduce_mean(tf.abs(z))
                
            elif self.regularization_type=='kl':
                regularization_loss = tf.reduce_mean(keras.losses.kl_divergence(data_y, reconstruction))
                
            else:
                raise ValueError("Regularization loss must be either 'L1' or 'kl' " )
#             weight=1/250


            #Print the losses
#             pdb.set_trace()
            total_loss = reconstruction_loss + self.weight*regularization_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)






        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "regularization_loss": self.regularization_loss_tracker.result()
        }

def plot_latent_space(combined_data,weight, BASE_PATCH_WIDTH,regularization_type='kl',recon_type='bce', n=30, figsize=15):
    
    '''
    
    display a n*n 2D manifold of cell images
    
    Args:
    -----
    combined_data: The image dataset.
    
    weight: stregnth of the regularization loss (L1 or KL).
    
    BASE_PATCH_WIDTH: size of the subimage.
    
    n: (optional) grid size.
    
    figsize: (optional) size of the figure.
      
    
    returns:
    --------
    Returns figure with a 2D manifold of cell images
    
    '''
    
    
    encoder,decoder=autoencoder(2,1,BASE_PATCH_WIDTH,summary='no')
    
    
    vae = VAE(encoder, decoder,weight,regularization_type,recon_type)
    vae.compile(optimizer='adam')
    vae.fit(combined_data, epochs=100, batch_size=128,verbose=0)  
    
    scale = 1.0
    figure = np.zeros((BASE_PATCH_WIDTH * n, BASE_PATCH_WIDTH * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(BASE_PATCH_WIDTH, BASE_PATCH_WIDTH)
            figure[
                i * BASE_PATCH_WIDTH : (i + 1) * BASE_PATCH_WIDTH,
                j * BASE_PATCH_WIDTH : (j + 1) * BASE_PATCH_WIDTH,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = BASE_PATCH_WIDTH // 2
    end_range = n * BASE_PATCH_WIDTH + start_range
    pixel_range = np.arange(start_range, end_range, BASE_PATCH_WIDTH)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

    
def plot_RAE_latent_space(combined_data,weight, BASE_PATCH_WIDTH,regularization_type='kl',recon_type='bce', n=30, figsize=15):
    
    '''
    
    display a n*n 2D manifold of cell images
    
    Args:
    -----
    combined_data: The image dataset.
    
    weight: stregnth of the regularization loss (L1 or KL).
    
    BASE_PATCH_WIDTH: size of the subimage.
    
    n: (optional) grid size.
    
    figsize: (optional) size of the figure.
      
    
    returns:
    --------
    Returns figure with a 2D manifold of cell images
    
    '''
    
    
    encoder,decoder=autoencoder(2,1,BASE_PATCH_WIDTH,summary='no')
    
    
    vae = Short_VAE(encoder, decoder,weight,regularization_type,recon_type)
    vae.compile(optimizer='adam')
    vae.fit(combined_data, epochs=100, batch_size=128,verbose=0)  
    
    scale = 1.0
    figure = np.zeros((BASE_PATCH_WIDTH * n, BASE_PATCH_WIDTH * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(BASE_PATCH_WIDTH, BASE_PATCH_WIDTH)
            figure[
                i * BASE_PATCH_WIDTH : (i + 1) * BASE_PATCH_WIDTH,
                j * BASE_PATCH_WIDTH : (j + 1) * BASE_PATCH_WIDTH,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = BASE_PATCH_WIDTH // 2
    end_range = n * BASE_PATCH_WIDTH + start_range
    pixel_range = np.arange(start_range, end_range, BASE_PATCH_WIDTH)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
  









    












    




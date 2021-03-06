
import numpy as np

from numpy import matlib
import time
from scipy.stats import norm
import random
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from matplotlib import rc

def _format_axes(ax, **kwargs):
    
#     rc('font', family = 'serif')
    
    # Set border
    border = True
    if 'border' in kwargs:
        border = kwargs['border']
        
    if border:
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    # Format label and tick fonts
    tickwidth = 1
    if border == False:
        tickwidth = 2
    
    ax.xaxis.label.set_size(12)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_size(12)
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=12, labelcolor = 'black')
    
    return ax


def soft_clustering_weights(data, cluster_centres, **kwargs):
    
    """
    arguments:
    A function to calculate the weights for soft k-means clustering
    data: numpy array,. Features arranged across the columns with each row being a different data point
    cluster_centres: numpy array of cluster centres. kmeans.cluster_centres_ is generally passed
    parameters: m - keyword argument, fuzziness/softness of the clustering. m = 2 is set as default
    """
    
    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    if 'm' in kwargs:
        m = kwargs['m']
    
    #number of clusters
    Nclusters = cluster_centres.shape[0]
    #number of data points
    Ndp = data.shape[0]
    #number of features
    Nfeatures = data.shape[1]

    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:,i] = np.sum((data-np.matlib.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)
    
   
    # Denominator of the weight:
    invWeight = EuclidDist**(2/(m-1))*np.matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
    Weight = 1./invWeight
    
    return Weight


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Definitions, paths and functions for the segmenter and annotator
"""


# %matplotlib inline
from IPython.display import clear_output

import sys
import os
import numpy as np
import pandas as pd
import glob
import h5py
import time
from distutils.dir_util import copy_tree
from shutil import copyfile, copy, copy2


from scipy.ndimage.filters import median_filter

import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tiff


from matplotlib_scalebar.scalebar import ScaleBar


from ipywidgets import interact
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from IPython.display import display, update_display
from ipyfilechooser import FileChooser
from skimage.io import imread

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
from skimage.filters import threshold_otsu, threshold_mean, threshold_local
from skimage import measure

import seaborn as sns




import sys
if '../' not in sys.path:
    sys.path.append('../')

# from tools.misc.Utils import CenterSampling, ClusterAnalysis
# from tools.misc.patches2d import Patches as Patches2D
# import tools.neural_nets.xrf_autoencoders
# from tools.neural_nets.xrf_autoencoders import *

from tools.datasets.xrf_datasets import *
from tools.misc.Utils import *


# To plot bounding box Warning name!
from matplotlib import patches


from openTSNE import TSNE
from sklearn.decomposition import PCA





#PATH DEFINITIONS
# Set the directory where the FILEChooser widget will open up to read .h5 files contating xrf images

annot_dir='annotated_XRF'
base__dir_path=os.path.join(os.path.join(os.path.dirname(os.getcwd()),annot_dir), 'raw_cells')
h5_dir = base__dir_path
default_path = h5_dir 

#OTHER DEFINITIONS
global VALUE_OFFSET, NORM_CH

# NORM_CH='US_IC'
# VALUE_OFFSET=1e-12

#FIGURE DEFINITIONS

global CROSS_HAIR_SIZE, SCALE_UNIT_FACTOR, FIGSIZE, cbar_position1,cbar_position2,cbar_position3
CROSS_HAIR_SIZE=15

SCALE_UNIT_FACTOR=0.000001
DEFAULT_RESOLUTION_CELL = 0.33


FIGSIZE=(9, 2.8)
SMALL_FIGSIZE=(8, 2.4)

#For adjusting colorbars in segmenter
cbar_position1=[0.055, 0.11, 0.255, 0.01]
cbar_position2=[0.38, 0.11, 0.255, 0.01]
cbar_position3=[0.71, 0.11, 0.255, 0.01]

#For adjusting colorbars in annotator
cbar_position1_annot=[0.32, 0.125, 0.01, 0.755]
cbar_position2_annot=[0.645, 0.125, 0.01, 0.755]
cbar_position3_annot=[0.965, 0.125, 0.01, 0.755]




def text_width(wd):
    return Layout(width = "%ipx"%wd)


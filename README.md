[![DOCS](https://readthedocs.org/projects/docs/badge/?version=latest)](https://roi-finder.readthedocs.io/en/latest/?badge=latest)

# XRF-ROI-Finder: Machine learning to guide region-of-interest scanning for X-ray fluorescence microscopy 

Software for intelligently guiding X-ray fluorescence scans to regions of interest using a suite of ML-based clustering algorithms. 


<p align="center">
  <img width="500" src="imgs/ROI-Finder.png">
</p>

## Installation

1. git clone https://github.com/arshadzahangirchowdhury/ROI-Finder.git
2. Install the packages via the AI_XRF_env.yml file.

## Instructions

1. The GUI based example workflows contains a segmenter, an annotator and a recommender tool.
2. Segmenter tool is used to select .h5 files containing XRF images and extract region of interests (cells).
3. Annotator tool is used to bin the extracted cells in two groups, accepts or rejects corresponding to alive or dead cells.

## Segmenter

The segmenter workflow is designed to identify and explore the parameters which affect the conversion process of images to binary images.

## Annotator

The annotator tool allows the user to bin the XRF images into two categories called "accepts" and "rejects". These two categories can correspond to "live" and "dead" bacterial cells respectively. The user can preview the extracted cells from the segmenter workflow and then use the buttons to annotate and bin the data into two groups. The annotated data is stored to the user's local hard drive inside the annotated_XRF folder. This directory must not be renamed or moved.

## Recommender

The recommender tool allows the user to select an AI method, based on which recommendations are given to the user based on bacterial cells which are similar to the selected cell.

## Intended user
The software is being developed for the users of the Bionanoprobe at the 9-ID beamline of the Advanced Photon Source (APS).  


# Acknowledgments



## Developers

M Arshad Zahangir Chowdhury, DSL, Argonne National Laboratory

Aniket Tekawade, DSL, Argonne National Laboratory

## Data and Ideas

Si Chen, XSD, Argonne National Laboratory

Grace Luo, XSD, Argonne National Laboratory

Zhengchun Liu, DSL, Argonne National Laboratory

Kiwon Ok, Michigan State University

Thomas O'Halloran, Michigan State University

Barry Lai, XSD, Argonne National Laboratory

Rajkumar Kettimuthu, DSL, Argonne National Laboratory


## Funding

This project was funded by a Laboratory Directed Research and Development (LDRD) grant from Argonne National Laboratory and  National Institutes of Health grants GM038784 and
P41GM181350 (to TVO)

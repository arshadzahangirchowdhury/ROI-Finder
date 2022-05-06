#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: arshadzahangirchowdhury
"""

from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='roifinder',
    url='https://github.com/arshadzahangirchowdhury/ROI-Finder',
    author='M Arshad Zahangir Chowdhury',
    author_email='arshad.zahangir.bd@gmail.com',
    # Needed to actually package something
    packages=find_packages(exclude=['test']),
#     packages= ['roifinder', 'roifinder.misc', 'roifinder.src'],
    # Needed for dependencies
    install_requires=['numpy', 'pandas', 'scipy', 'h5py', 'matplotlib', 'opencv-python',\
                      'scikit-image','scikit-learn', 'seaborn' ,'ipython'],
    version=open('VERSION').read().strip(),
#     version="1.0",
    license='BSD',
    description='XRF-ROI-Finder: Machine learning to guide region-of-interest scanning for X-ray fluoroscence microscopy',
#     long_description=open('README.md').read(),
)
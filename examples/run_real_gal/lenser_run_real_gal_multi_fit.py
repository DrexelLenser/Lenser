import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits



"""
Module: lenser_run_real_gal_multi_fit
.. synopsis: Imports a real galaxy in multiple image bands and then runs it through Lenser in multi-fit mode
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. This script takes as an input the path to a galaxy image, in multiple bands, and reads in all relevant .FITS files.
.. In multi-band mode, `Lenser` fits a single model to multiple postage stamps, each representing an exposure of a 
   single galaxy in a particular band.  For example, one might do a simultaneous fit to the (r,i,z) bands in 
   multi-fit mode, rather than just fitting the r band in single-fit mode.
"""



# Read in images from FITS file. We will choose a galaxy from the COSMOS catalogue
# .. Band 1: F814W
# .. .. Specify path to science image
path_to_image = '../Catalogues/COSMOS/Images_F814W/COSMOS_5168_F814W.fits'
# .. .. Get science image, noisemap, segmentation mask (uberseg by default),
#       psfmap, and background mask from lenser_fits:
f = FITS(path_to_image)
dat_F814W = f.get_FITS('data')
rms_F814W = f.get_FITS('noise')
seg_F814W = f.get_FITS('segmask')
psf_F814W = f.get_FITS('psf') 
bg_F814W = f.get_FITS('bgmask')
# .. .. Get name of object and band from path_to_image (can be overridden)
image_name_F814W = path_to_image.split('/')[-1].split('.')[0]
# .. Band 2: F606W
path_to_image = '../Catalogues/COSMOS/Images_F606W/COSMOS_5168_F606W.fits'
f = FITS(path_to_image)
dat_F606W = f.get_FITS('data')
rms_F606W = f.get_FITS('noise')
seg_F606W = f.get_FITS('segmask')
psf_F606W = f.get_FITS('psf') 
bg_F606W = f.get_FITS('bgmask')
image_name_F606W = path_to_image.split('/')[-1].split('.')[0]
# .. Band 3: F125W
path_to_image = '../Catalogues/COSMOS/Images_F125W/COSMOS_5168_F125W.fits'
f = FITS(path_to_image)
dat_F125W = f.get_FITS('data')
rms_F125W = f.get_FITS('noise')
seg_F125W = f.get_FITS('segmask')
psf_F125W = f.get_FITS('psf') 
bg_F125W = f.get_FITS('bgmask')
image_name_F125W = path_to_image.split('/')[-1].split('.')[0]

# Create a MultiImage instance
# .. Elliptical mask is generated and background subtracted by default
myMultiImage = MultiImage(namelist = [image_name_F814W, image_name_F606W, image_name_F125W],
                          datalist = [dat_F814W, dat_F606W, dat_F125W],
                          noiselist = [rms_F814W, rms_F606W, rms_F125W],
                          uberseglist = [seg_F814W, seg_F606W, seg_F125W],
                          psflist = [psf_F814W, psf_F606W, psf_F125W],
                          bgmasklist = [bg_F814W, bg_F606W, bg_F125W])

# Initialize AIM model
myModel = aimModel(myMultiImage = myMultiImage)

# Run local minimization
myModel.runLocalMinRoutine()

# Reset the parameters to their default values
myModel.empty()

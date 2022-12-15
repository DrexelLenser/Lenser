import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits



"""
Module: lenser_run_real_gal_single_fit
.. synopsis: Imports a real galaxy image and then runs it through Lenser in single-fit mode
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. This script takes as an input the path to a galaxy image and reads in all relevant .FITS files.
.. It generates an elliptical mask, finds and subtracts a background,
    and performs a chisquared minimzation using the aimModel().runLocalMin() function, 
    which models a galaxy by using the modified S\'ersic-type intensity profile of 
    [arXiv:2006.03506](https://arxiv.org/abs/2006.03506)
.. Plots are created for the real galaxy image, the mask, the noisemap, and the PSF (if provided)
.. A plot is created comparing the real galaxy image, the model, and the difference between
   the two.
"""



# Read in image from FITS file. We will choose a galaxy from the COSMOS catalogue
# .. Specify path to science image
path_to_image = '../Catalogues/COSMOS/Images_F814W/COSMOS_5168_F814W.fits'
# .. Get science image, noisemap, segmentation mask (uberseg by default),
#    psfmap, and background mask from lenser_fits:
f = FITS(path_to_image)
dat = f.get_FITS('data')
rms = f.get_FITS('noise')
seg = f.get_FITS('segmask')
psf = f.get_FITS('psf') 
bg = f.get_FITS('bgmask')
# .. Get name of object from path_to_image (can be overridden)
image_name = path_to_image.split('/')[-1].split('.')[0]

# Create an Image instance
myImage = Image(name = image_name, datamap = dat, noisemap = rms, ubersegmask = seg,
                psfmap = psf, bgmask = bg)
# .. Plot science image
myImage.plot(save=False, show=True)

# Generate elliptical mask and subtract background
myImage.generateEllipticalMask(subtractBackground=True)
# .. Plot the total mask (elliptical multiplied by seg)
myImage.plot(type='totalmask', save=False, show=True)

# Plot noisemap
myImage.plot(type='noise', save=False, show=True)

# Plot PSF
myImage.plot(type='psf', save=False, show=True)

# Initialize AIM model
myModel = aimModel(myImage)

# Run local minimization
myModel.runLocalMinRoutine()

# Plot the real galaxy image, the best-fit model, and the difference between the two
myModel.make_plot_compare(save=False, show=True)
# .. Zoom in for visual comparison
myModel.make_plot_compare(zoom=True, save=False, show=True)

# Reset the parameters to their default values
myModel.empty()

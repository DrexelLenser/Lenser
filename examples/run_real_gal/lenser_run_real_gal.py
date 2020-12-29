import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits



"""
Module: lenser_run_real_gal
.. synopsis: Imports a real galaxy image and then runs it through Lenser 
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. This script will import an image of a galaxy from one of the Catalogues in the Catalougues folder.
.. Plots are created for the real galaxy image, the elliptical mask, the noisemap, and the PSF (if provided)
.. A plot is created comparing the real galaxy image, the model, and the difference between
   the two.
"""

# Read in image from FITS file. We will choose a galaxy from the COSMOS catalogue
myImage=Image('../Catalogues/COSMOS/ima_r/COSMOS_2572_r.fits')
myImage.plot(save=True, show=True)

# Generate mask and plot it
myImage.generateMask(subtractBackground=True)
myImage.plot(type='mask', save=False, show=True)

# Plot noisemap
myImage.plot(type='noise', save=False, show=True)

# Plot PSF
myImage.plot(type='psf', save=False, show=True)

# Initialize AIM model
myModel = aimModel(myImage)

# Run local minimization
myModel.runLocalMinRoutine()

# Return 1sigma errors on parameters from chisquared best-fit
myModel.getParErrors()

# Check for a realistic fit
myModel.checkFit()

# Plot the real galaxy image, the best-fit model, and the difference between the two
myModel.make_plot_compare(save=False, show=True)
# Zoom in for visual comparison
myModel.make_plot_compare(zoom=True, save=False, show=True)

# Reset the parameters to their default values
myModel.empty()

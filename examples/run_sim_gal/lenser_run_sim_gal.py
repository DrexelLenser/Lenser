import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits



"""
Module: lenser_run_sim_gal
.. synopsis: Simulates a galaxy image and then runs it through Lenser 
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. One can use Lenser in order to simulate a postage stamp of a galaxy. In this case, the galaxy 
   itself is modeled using the modified S\eersic-type intensity profile, some sky background b is 
   added to the image, and randomly generated noise is added, such that each pixel i in the stamp 
   has a value given by:
     f_i = I_i + n_i ∗ numpy.random.normal(size=(N1, N2)) + b_i
   where the noisemap
     n_i = sqrt(n_{a,i}^2 + (n_{b,i} * sqrt(I_i))^2)
   where n_{a,i} is the sky noise and n_{b,i} * sqrt(I_i) is the Poisson noise.
.. For example, one could create a random Lens of the form
     myLens = Lens(psi2=[0,0,0],psi3=[0.001,-0.003,0.003,0.0002]) 
   to lens some galaxy
     myGalaxy = Galaxy(xc=0,yc=0,ns=0.75,rs=2.,q=3.5,phi=1*np.pi/6,galaxyLens=myLens)
.. This script demonstrates how to simulate such a postage stamp, and then export the datamap and
   noisemap as .FITS files to the working directory.
.. This script then reads in those .FITS files, generates a mask, finds and subtracts a background,
   and performs a chisquared minimzation using the aimModel().runLocalMin() function.
.. Plots are created for the simulated galaxy image, the elliptical mask, and the noisemap.
.. A plot is created comparing the simualted galaxy image, the model, and the difference between
   the two.
"""

# e.g. of a Lens:
myLens = Lens(psi2=[0,0,0],psi3=[0.01,-0.003,0.007,-0.003]) 

# Create a Galaxy object
myGalaxy = Galaxy(xc=0,yc=0,ns=1.,rs=2.,q=2.,phi=1.25,galaxyLens=myLens)

# Simulate a real galaxy image
myImage=myGalaxy.generateImage(100,100,lens=True,I0=1.e3,noise1=1.,noise2=0.1,background=0.,seed=0) 

# Save image to a FITS file
hdu = fits.PrimaryHDU(myImage.getMap())
hdu.writeto('Simulated_Galaxy.fits',overwrite=True)

# Save noisemap to a FITS file
hdu_noisemap = fits.PrimaryHDU(myImage.getMap(type='noise'))
hdu_noisemap.writeto('Simulated_Galaxy_rms.fits',overwrite=True)

# Reset the parameters to their default values
aimModel().empty

# Read in image from FITS file
myImage = Image('Simulated_Galaxy.fits')
myImage.plot(save=True)

# Initialize AIM model
myModel = aimModel(myImage)

# Generate mask and plot it
myImage.generateMask(subtractBackground=True)
myImage.plot(type='mask',save=True)

# Plot noisemap
myImage.plot(type='noise',save=True)

# Run local minimization
myModel.runLocalMinRoutine()

# Check for a realistic fit
myModel.checkFit()

# Return 1sigma errors on parameters from chisquared best-fit
myModel.getParErrors()

# Plot the simulated galaxy, the best-fit model, and the difference between the two
myModel.make_plot_compare(save=True)
# Zoom in for visual comparison
myModel.make_plot_compare(zoom=True,save=True)

# Reset the parameters to their default values
myModel.empty()

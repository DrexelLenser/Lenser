import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits
from scipy.special import gamma



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

# Choose which galaxy catalogue to mimic: COSMOS, EFIGI, or other.  If other is chosen,
#  you will need to specify the following parameters yourself: 
#  .. Postage stamp size
#  .. Galaxy size in pixels
#  .. Range of galaxy brightness, 
#  .. Stamp noise 
#  .. Sky background
catalogue_type = 'COSMOS'
#   COSMOS type catalogue
if catalogue_type == 'COSMOS':
    # Generate non-fit parameters.  
    # .. These values should be motivated to reflect actual data
    # .. Postage stamp size
    Nx = 150
    Ny = 150
    # .. Standard galaxy size (in pixels)
    a = 30
    # .. I0
    I0 = 5.
    # .. noise1 and noise2
    noise1 = 1.3e-3
    noise2 = 0.
    # .. Background
    background = 0.
#  EFIGI type catalogue
elif catalogue_type == 'EFIGI':
    # .. Postage stamp size
    Nx = 255
    Ny = 255
    # .. Standard galaxy size (in pixels)
    a = 30
    # .. I0
    I0 = 5.e4
    # .. noise1 and noise2
    noise1 = 2.
    gain = 4.75
    noise2 = 1/np.sqrt(gain)
    # .. Background
    background = 0.
else:
    print('Other was chosen for catalogue type.  User will specify catalogue-type parameters themselves.')

# Lensing parameters
# .. We will choose gamma1 and gamma2  and then get psi,ij.
#     We will set kappa = 0 (we can arbitrarily make this choice due to the 
#     mass-sheet degeneracy)
# .. kappa
kappa = 0.
# .. gamma1
gamma1 = (0.05)/np.sqrt(2)
# .. gamma2
gamma2 = (0.05)/np.sqrt(2)
# .. psi,11
psi11 = kappa + gamma1
# .. psi,12
psi12 = gamma2
# .. psi,22
psi22 = kappa - gamma1
# .. We have to be careful when generating the flexion, because not all of psi,ijj 
#     are independent from one another. We do the following:
#      (i).   Choose F1 and F2 
#      (ii).  Use F1 and F2 to calculate the angle of flexion, phi_F
#      (iii). Assume a particular analytic lens model, which in this case is a 
#             singular isothermal sphere (SIS).  This allows us to relate first and
#             section flexion in an analytic way.  We then use F1, F2, and phi_F to 
#             get G1 and G2 
#      (iv).  Use F1, F2, G1, and G2 to get psi,ijk
# .. F1
F1 = (1.e-3)/np.sqrt(2)
# .. F2
F2 = (1.e-3)/np.sqrt(2)
# .. phi_F
# .. .. angle of flexion
phi_F = np.arctan2(F2,F1)
# .. G1
G1 = -((3*np.cos(3*phi_F))/np.cos(phi_F))*F1
# .. G2
G2 = -((3*np.sin(3*phi_F))/np.sin(phi_F))*F2
# .. psi,111
psi111 = (1./2.)*(3.*F1 + G1)
# .. psi,112
psi112 = (1./2.)*(F2 + G2)
# .. psi,122
psi122 = (1./2.)*(F1 - G1)
# .. psi,222
psi222 = (1./2.)*(3.*F2 - G2)

# Shape parameters
# .. Centroid
# .. .. Dither the centroid (make it slightly off-center)
xc = 0.5
yc = 0.5
# .. ns
ns = 2.5
# .. phi
phi = np.pi/6
# .. q
# .. .. Axis ratio will be a function of both intrinsic ellipticity and shear
# .. .. We choose intrinsic ellipticity to have a magnitude of 0.2 (Schneider 1996)
eps_s = 0.2
eps_s1, eps_s2 = eps_s*np.cos(2.*phi), eps_s*np.sin(2.*phi)
eps1 = eps_s1 + gamma1
eps2 = eps_s2 + gamma2
eps = np.sqrt(eps1**2. + eps2**2.)
q_obs = (1+abs(eps))/(1-abs(eps))
q = (1+abs(eps_s))/(1-abs(eps_s))
# .. rs
rs = a/(np.sqrt(((1+q_obs**2.)/2)))*np.sqrt(gamma(2.*ns)/gamma(4.*ns))

# Create lens
myLens = Lens(psi2=[psi11,psi12,psi22],psi3=[psi111,psi112,psi122,psi222])

# Create a Galaxy object
myGalaxy = Galaxy(xc,yc,ns,rs,q,phi,galaxyLens=myLens)

# Simulate a real galaxy image
myImage=myGalaxy.generateImage(nx=Nx,ny=Ny,lens=True,I0=I0,noise1=noise1,noise2=noise2,background=background,seed=0) 

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
myImage.plot(show=True)

# Initialize AIM model
myModel = aimModel(myImage)

# Generate mask and plot it
myImage.generateMask(subtractBackground=True)
myImage.plot(type='mask',show=True)

# Plot noisemap
myImage.plot(type='noise',show=True)

# Run local minimization
myModel.runLocalMinRoutine()

# Check for a realistic fit
myModel.checkFit()

# Return 1sigma errors on parameters from chisquared best-fit
myModel.getParErrors()

# Plot the simulated galaxy, the best-fit model, and the difference between the two
myModel.make_plot_compare(show=True)
# Zoom in for visual comparison
myModel.make_plot_compare(zoom=True,show=True)

# Reset the parameters to their default values
myModel.empty()

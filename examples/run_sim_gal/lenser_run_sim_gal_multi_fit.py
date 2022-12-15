import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Module: lenser_run_sim_gal_multi_fit
.. synopsis: Simulates a galaxy image in multiple bands and then runs it through Lenser in multi-fit mode 
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. Across different bands, a galaxy will vary (slightly) by morphology.  To simulate this, this script chooses
   different values of ns, rs, and I0 for each band, while keeping all other parameters fixed.  
   It is shown that the constant lensing parameters are fit well at the cost of fitting ns and rs poorly.
"""

# Postage stamp size (motivated by COSMOS catalogue)
Nx = 150
Ny = 150

# Make sky-doimnated noise
noise1 = 1e-3 # Motivated by typical galaxy in COSMOS catalogue
noise2 = 0
# Make background-subtracted
background=0

# Common parameters across all epochs, bands:
# .. Motivated by the fit parameters from a galaxy in the COSMOS catalogue
xc = -2.86
yc = -1.54
q = 1.94
phi = 1.25
psi11 = 0.0 
psi12 = 0.0 
psi22 = 0.0 
psi111 = 0.010
psi112 = -0.0037 
psi122 = 0.0069
psi222 = -0.0027

# Band-dependent parameters: ns, rs, and I0.
#  Motivated by the fit paramters from a galaxy in the COSMOS catalogue in the F814W, F606W, and F125W filters
# .. Band 1:
ns_b1 = 1.25
rs_b1 = 1.86
I0_b1 = 0.24
# .. Band 2:
ns_b2 = 1.28
rs_b2 = 1.56
I0_b2 = 0.27
# .. Band 3:
ns_b3 = 1.15
rs_b3 = 1.28
I0_b3 = 1.21

# Collect parameters
pars_input_b1 = [xc,yc,ns_b1,rs_b1,q,phi,psi111,psi112,psi122,psi222]
pars_input_b2 = [xc,yc,ns_b2,rs_b2,q,phi,psi111,psi112,psi122,psi222]
pars_input_b3 = [xc,yc,ns_b3,rs_b3,q,phi,psi111,psi112,psi122,psi222]

# Create lens
myLens = Lens(psi3=[psi111,psi112,psi122,psi222])

# Create a Galaxy object for each band
myGalaxy_b1 = Galaxy(xc,yc,ns_b1,rs_b1,q,phi,galaxyLens=myLens)
myGalaxy_b2 = Galaxy(xc,yc,ns_b2,rs_b2,q,phi,galaxyLens=myLens)
myGalaxy_b3 = Galaxy(xc,yc,ns_b3,rs_b3,q,phi,galaxyLens=myLens)

# Simulate a real galaxy image for each band
myImage_b1 = myGalaxy_b1.generateImage(nx=Nx,ny=Ny,lens=True,I0=I0_b1,noise1=noise1,noise2=noise2,background=background,seed=0) 
myImage_b2 = myGalaxy_b2.generateImage(nx=Nx,ny=Ny,lens=True,I0=I0_b2,noise1=noise1,noise2=noise2,background=background,seed=0) 
myImage_b3 = myGalaxy_b3.generateImage(nx=Nx,ny=Ny,lens=True,I0=I0_b3,noise1=noise1,noise2=noise2,background=background,seed=0) 

# Save each image to a FITS file
hdu_b1 = fits.PrimaryHDU(myImage_b1.getMap('data'))
hdu_b1.writeto('sim_gal_band1.fits',overwrite=True)
hdu_b2 = fits.PrimaryHDU(myImage_b2.getMap('data'))
hdu_b2.writeto('sim_gal_band2.fits',overwrite=True)
hdu_b3 = fits.PrimaryHDU(myImage_b3.getMap('data'))
hdu_b3.writeto('sim_gal_band3.fits',overwrite=True)

# Save noisemap to a FITS file
hdu_noisemap = fits.PrimaryHDU(noise1*np.ones((Nx,Ny)))
hdu_noisemap.writeto('sim_gal_band1_rms.fits',overwrite=True)
hdu_noisemap.writeto('sim_gal_band2_rms.fits',overwrite=True)
hdu_noisemap.writeto('sim_gal_band3_rms.fits',overwrite=True)

# Read in images from FITS file
# .. b1
path_to_image = 'sim_gal_band1.fits'
f = FITS(path_to_image)
dat_b1 = f.get_FITS('data')
rms_b1 = f.get_FITS('noise')
seg_b1 = f.get_FITS('segmask')
bg_b1 = f.get_FITS('bgmask')
# .. b2
path_to_image = 'sim_gal_band2.fits'
f = FITS(path_to_image)
dat_b2 = f.get_FITS('data')
rms_b2 = f.get_FITS('noise')
seg_b2 = f.get_FITS('segmask')
bg_b2 = f.get_FITS('bgmask')
# .. b3
path_to_image = 'sim_gal_band3.fits'
f = FITS(path_to_image)
dat_b3 = f.get_FITS('data')
rms_b3 = f.get_FITS('noise')
seg_b3 = f.get_FITS('segmask')
bg_b3 = f.get_FITS('bgmask')

# Run each band individually in single-fit mode:
# .. b1
print('Band 1 single-fit mode:')
print('\n')
myImage_b1 = Image(name = 'sim_gal_band1', datamap = dat_b1, noisemap = rms_b1, segmask = seg_b1, bgmask = bg_b1)
#myImage_b1.plot(show=True)
myModel_b1 = aimModel(myImage_b1)
myImage_b1.generateEllipticalMask(subtractBackground=True)
#myImage_b1.plot(type='totalmask',show=True)
#myImage_b1.plot(type='noise',show=True)
myModel_b1.runLocalMinRoutine()
#myModel_b1.make_plot_compare(show=True)
#myModel_b1.make_plot_compare(zoom=True,show=True)
pars_b1 = myModel_b1.parsWrapper()[np.where(myModel_b1.doFlags==1)]
pars_err_b1 = myModel_b1.parsErrors[np.where(myModel_b1.doFlags==1)]
myModel_b1.empty()
# .. b2
print('\n')
print('Band 2 single-fit mode:')
print('\n')
myImage_b2 = Image(name = 'sim_gal_band1', datamap = dat_b2, noisemap = rms_b2, segmask = seg_b2, bgmask = bg_b2)
#myImage_b2.plot(show=True)
myModel_b2 = aimModel(myImage_b2)
myImage_b2.generateEllipticalMask(subtractBackground=True)
#myImage_b2.plot(type='totalmask',show=True)
#myImage_b2.plot(type='noise',show=True)
myModel_b2.runLocalMinRoutine()
#myModel_b2.make_plot_compare(show=True)
#myModel_b2.make_plot_compare(zoom=True,show=True)
pars_b2 = myModel_b2.parsWrapper()[np.where(myModel_b2.doFlags==1)]
pars_err_b2 = myModel_b2.parsErrors[np.where(myModel_b2.doFlags==1)]
myModel_b2.empty()
# .. b3
print('\n')
print('Band 3 single-fit mode:')
print('\n')
myImage_b3 = Image(name = 'sim_gal_band1', datamap = dat_b3, noisemap = rms_b3, segmask = seg_b3, bgmask = bg_b3)
#myImage_b3.plot(show=True)
myModel_b3 = aimModel(myImage_b3)
myImage_b3.generateEllipticalMask(subtractBackground=True)
#myImage_b3.plot(type='totalmask',show=True)
#myImage_b3.plot(type='noise',show=True)
myModel_b3.runLocalMinRoutine()
#myModel_b3.make_plot_compare(show=True)
#myModel_b3.make_plot_compare(zoom=True,show=True)
pars_b3 = myModel_b3.parsWrapper()[np.where(myModel_b3.doFlags==1)]
pars_err_b3 = myModel_b3.parsErrors[np.where(myModel_b3.doFlags==1)]
myModel_b3.empty()

# Do multi-band fit
print('\n')
print('Multi-fit mode:')
print('\n')
myMultiImage = MultiImage(namelist = ['sim_gal_band1', 'sim_gal_band2', 'sim_gal_band3'],
                          datalist = [dat_b1, dat_b2, dat_b3],
                          noiselist = [rms_b1, rms_b2, rms_b3],
                          seglist = [seg_b1, seg_b2, seg_b3],
                          bgmasklist = [bg_b1,bg_b2,bg_b3])
myModel = aimModel(myMultiImage = myMultiImage)
myModel.runLocalMinRoutine()
pars_multi_b = myModel.parsWrapper()[np.where(myModel.doFlags==1)]
pars_err_multi_b = myModel.parsErrors[np.where(myModel.doFlags==1)]

# Now create a plot for visualization purposes.
# Plot should be the input and reconstructed parameters for
#  each single-fit as well as the multi-fit
par_names =  [r'$\theta_0^1$',
              r'$\theta_0^2$',
              r'$n_s$',
              r'$\theta_s$',
              r'$q$',
              r'$\phi$',
              r'$\psi,_{111}$',
              r'$\psi,_{112}$',
              r'$\psi,_{122}$',
              r'$\psi,_{222}$']
f, ax = plt.subplots(1, 10, figsize=(15,5))
for i in range(10):
  if i!=2 and i!=3:
      ax[i].scatter(0,pars_input_b1[i], color='k', marker = 'X', label=r'${\rm Input~pars}$')
      ax[i].scatter(0,pars_input_b2[i], color='k', marker = 'X')
      ax[i].scatter(0,pars_input_b3[i], color='k', marker = 'X')
      ax[i].errorbar(1,pars_b1[i], yerr = pars_err_b1[i], fmt = 'o', label=r'${\rm Band}\,\,1$')
      ax[i].errorbar(2,pars_b2[i], yerr = pars_err_b2[i], fmt = 'o', label=r'${\rm Band}\,\,2$')
      ax[i].errorbar(3,pars_b3[i], yerr = pars_err_b3[i], fmt = 'o', label=r'${\rm Band}\,\,3$')
      ax[i].errorbar(4,pars_multi_b[i], yerr = pars_err_multi_b[i], fmt='o', label=r'${\rm Multi-fit}$')
      ax[i].set_xlabel(par_names[i])
      ax[i].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
      ax[i].set_xlim(-0.5,4.5)
  else:
      ax[i].scatter(0,pars_input_b1[i], color='blue', marker = 'X')
      ax[i].scatter(0,pars_input_b2[i], color='orange', marker = 'X')
      ax[i].scatter(0,pars_input_b3[i], color='green', marker = 'X')
      ax[i].errorbar(1,pars_b1[i], yerr = pars_err_b1[i], fmt = 'o')
      ax[i].errorbar(2,pars_b2[i], yerr = pars_err_b2[i], fmt = 'o')
      ax[i].errorbar(3,pars_b3[i], yerr = pars_err_b3[i], fmt = 'o')
      ax[i].errorbar(4,pars_multi_b[i], yerr = pars_err_multi_b[i], fmt='o', label=r'${\rm Multi-fit}$')
      ax[i].set_xlabel(par_names[i])
      ax[i].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
      ax[i].set_xlim(-0.5,4.5)
legend=ax[-1].legend(framealpha=0, fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()




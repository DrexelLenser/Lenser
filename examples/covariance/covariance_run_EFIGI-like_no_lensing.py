from covariance import Covariance
import numpy as np
from scipy.special import gamma

"""Covariance run 2: EFIGI-like with no lensing fields"""
# Generate non-fit parameters.  
# .. These values should be motivated to reflect actual data
# .. Postage stamp size
Nx = 255
Ny = 255
# .. Standard galaxy size (in pixels)
a = 100.
# .. I0
I0 = 5.e4
# .. noise1 and noise2
noise1 = 2.
gain = 4.75
noise2 = 1/np.sqrt(gain)
# .. Background
background = 0.
# Set lensing fields to zero
psi2 = [0,0,0]
psi3 = [0,0,0,0]
# Shape parameters
# .. Centroid (will be dithered within a pixel in Covariance)
xc = 0
yc = 0
# .. ns
ns = 4.
# .. phi
phi = np.pi/6
# .. q
# .. .. We choose intrinsic ellipticity to have a magnitude of 0.2 (Schneider 1996)
eps_s = 0.2
q = (1+abs(eps_s))/(1-abs(eps_s))
# .. rs
rs = a/(np.sqrt(((1+q**2.)/2)))*np.sqrt(gamma(2.*ns)/gamma(4.*ns))
# Gather list of fiducial parameter values 
fid_params = np.array((0.5, 0.5, # Centroid dithered from 0 to 1, so the fiducial value is trivially 0.5
                       ns, rs,
                       q, phi,
                       psi111, psi112, psi122, psi222))
# Run Covariance
Cov2 = Covariance(Nx=Nx, Ny=Ny,
                  xc=xc, yc=yc, ns=ns, rs=rs, q=q, phi=phi,
                  psi2=psi2, psi3=psi3,
                  marg=np.array((1,1,1,1,1,1,0,0,0,1,1,1,1)),
                  I0=I0, noise1=noise1, noise2=noise2, background=background,
                  N_iter=1000,
                  fid_params=fid_params,
                  stamp_col_label='EFIGI-like_no_lensing_eps_0.5')
# Simulate the stamp collection
Cov2.simulateGals()
# Run Lenser on this stamp collection
Cov2.lenserRun()
# Compute the covariance matrix for this stamp collection
Cov2_mat = Cov2.computeCovMat()
# Compute the 1-sigma uncertainty on each parameter
print(np.round(Cov2.error(Cov2_mat),7))
# Plot
Cov2.plot_error_matrix(Cov2_mat)

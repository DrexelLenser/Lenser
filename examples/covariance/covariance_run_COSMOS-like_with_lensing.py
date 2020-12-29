from covariance import Covariance
import numpy as np
from scipy.special import gamma

"""Covariance run 3: COSMOS-like with lensing fields"""
# Generate non-fit parameters.  
# .. These values should be motivated to reflect actual data
# .. Postage stamp size
Nx = 150
Ny = 150
# .. Standard galaxy size (in pixels)
a = 30.
# .. I0
I0 = 5.
# .. noise1 and noise2
noise1 = 1.3e-3
noise2 = 0.
# .. Background
background = 0.
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
# .. Centroid (will be dithered within a pixel in Covariance)
xc = 0
yc = 0
# .. ns
ns = 2.5
# .. phi
phi = np.pi/6
# .. q
# .. .. Axis ratio will be a function of both intrinsic ellipticity and shear
# .. .. We choose intrinsic ellipticity to have a magnitude of 0.2 (Schneider 1996)
eps_s = 0.2
q = (1+abs(eps_s))/(1-abs(eps_s))
# .. .. Let us calculate the "observed q" i.e. the q that Lenser will reconstruct.
#        This q will be different from the above q, because nonzero shear will add
#        to the intrinsic ellipticity.
# .. .. .. Get the components of the intrinsic ellipticity. 
eps_s1, eps_s2 = eps_s*np.cos(2.*phi), eps_s*np.sin(2.*phi)
# .. .. .. Approximate observed ellipticity as eps = eps_s + gamma
eps1 = eps_s1 + gamma1
eps2 = eps_s2 + gamma2
eps = np.sqrt(eps1**2. + eps2**2.)
# .. .. .. Get observed q
q_obs = (1+abs(eps))/(1-abs(eps))
# .. .. Now let us get the "observed" phi.  By the same token as for q, the orientation
#        angle will be different from the intrinsic one in the presence of nonzero shear
phi_obs = np.arctan2(eps2,eps1)/2
# .. rs
rs = a/(np.sqrt(((1+q_obs**2.)/2)))*np.sqrt(gamma(2.*ns)/gamma(4.*ns))
# Gather list of fiducial parameter values 
#   will differ from actual input parameters in presence of nonzero shear 
#   i.e. q and phi change when shear is introduced
fid_params = np.array((0.5, 0.5, # Centroid dithered from 0 to 1, so the fiducial value is trivially 0.5
                       ns, rs,
                       q_obs, phi_obs,
                       psi111, psi112, psi122, psi222))
# Run Covariance
Cov3 = Covariance(Nx=Nx, Ny=Ny,
                  xc=xc, yc=yc, ns=ns, rs=rs, q=q, phi=phi,
                  psi2=[psi11,psi12,psi22], 
                  psi3=[psi111,psi112,psi122,psi222],
                  marg=np.array((1,1,1,1,0,0,0,0,0,1,1,1,1)),
                  I0=I0, noise1=noise1, noise2=noise2, background=background,
                  N_iter=1000,
                  fid_params=fid_params,
                  stamp_col_label='COSMOS-like_with_lensing_ns_2.5_a_50')
# Simulate the stamp collection
Cov3.simulateGals()
# Run Lenser on this stamp collection
Cov3.lenserRun()
# Compute the covariance matrix for this stamp collection
Cov3_mat = Cov3.computeCovMat()
# Compute the 1-sigma uncertainty on each parameter
print(np.round(Cov3.error(Cov3_mat),7))
# Plot
Cov3.plot_error_matrix(Cov3_mat)

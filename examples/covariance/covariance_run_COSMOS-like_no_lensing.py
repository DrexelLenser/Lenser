from covariance import Covariance
import numpy as np
from scipy.special import gamma

"""Covariance run 1: COSMOS-like with no lensing fields"""
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
# Set lensing fields to zero
psi2 = [0,0,0]
psi3 = [0,0,0,0]
# Shape parameters
# .. Centroid (will be dithered within a pixel in Covariance)
xc = 0
yc = 0
# .. ns
ns = 2.5
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
Cov1 = Covariance(Nx=Nx, Ny=Ny,
                  xc=xc, yc=yc, ns=ns, rs=rs, q=q, phi=phi,
                  psi2=psi2, psi3=psi3,
                  marg=np.array((1,1,1,1,1,1,0,0,0,1,1,1,1)),
                  I0=I0, noise1=noise1, noise2=noise2, background=background,
                  N_iter=1000,
                  fid_params=fid_params,
                  stamp_col_label='COSMOS-like_no_lensing_ns_2.5_a_50')
# Simulate the stamp collection
Cov1.simulateGals()
# Run Lenser on this stamp collection
Cov1.lenserRun()
# Compute the covariance matrix for this stamp collection
Cov1_mat = Cov1.computeCovMat()
# Compute the 1-sigma uncertainty on each parameter
print(np.round(Cov1.error(Cov1_mat),7))
# Plot
Cov1.plot_error_matrix(Cov1_mat)





































"""
# Create an instance of the Covariance class
# We would like to keep all default values, expect we will
#  add a flat noisemap with a value of I0/1e4
Cov1=Covariance(noise1=0.1, marg=np.array((1,1,1,1,1,1,0,0,0,1,1,1,1)), catalogue_label='noise1_I0e-4')

# Simulate the catalogue with this noise
Cov1.simulateGals()

# Run Lenser on this catalogue
Cov1.lenserRun()

# Compute the covariance matrix for this catalogue
Cov1_mat = Cov1.computeCovMat()

# Compute the 1-sigma uncertainty on each parameter
print(np.round(Cov1.error(Cov1_mat),7))

# Plot
Cov1.plot_error_matrix(Cov1_mat)


# Now let us repeat this process but for a flat 
#  noisemap with double the previous noise, 
#  i.e. a value of 2*I0/1e4
Cov2=Covariance(noise1=0.2, marg=np.array((1,1,1,1,1,1,0,0,0,1,1,1,1)), catalogue_label='noise1_2I0e-4')

# Simulate the catalogue with this noise
Cov2.simulateGals()

# Run Lenser on this catalogue
Cov2.lenserRun()

# Compute the covariance matrix for this catalogue
Cov2_mat = Cov2.computeCovMat()

# Compute the 1-sigma uncertainty on each parameter
print(np.round(Cov2.error(Cov2_mat),7))

# Plot
Cov2.plot_error_matrix(Cov2_mat)



# Let's plot both of these covariance matrices 
#  together for combined visualization
Cov1_vs_Cov2=Covariance(marg=np.array((1,1,1,1,1,1,0,0,0,1,1,1,1)))
Cov1_vs_Cov2_mats=Cov1_vs_Cov2.covariance_array(Cov1_mat,Cov2_mat)
Cov1_vs_Cov2.plot_error_matrix_combined(Cov1_vs_Cov2_mats, filename='about/simulated_stamp_collections_noise1_I0e-4_and_noise1_2I0e-4_combined', labels=[r'$I_0/10^4$',r'$2I_0/10^4$'])


# Create an instance of the Covariance class
# We would like to keep all default values, expect we will
#  add a flat noisemap with a value of I0/1e4
Cov3=Covariance(noise1=0.1, q=2.5, phi=2., psi3=[0.001,0.002,0.003,0.004],marg=[1,1,1,1,1,1,0,0,0,1,1,1,1],catalogue_label='noise1_I0e-4_phi_2_q_2.5_psi3_0.001_0.002_0.003_0.004')
#Cov3=Covariance(noise1=0.1, q=2.5, phi=0, psi3=[0,0,0,0],catalogue_label='noise1_I0e-4_phi_0_q_1_psiijk_0')

# Simulate the catalogue with this noise
Cov3.simulateGals(overwrite=True)

# Run Lenser on this catalogue
Cov3.lenserRun(overwrite=True)

# Compute the covariance matrix for this catalogue
Cov3_mat = Cov3.computeCovMat()

# Compute the 1-sigma uncertainty on each parameter
print(np.round(Cov3.error(Cov3_mat),7))

# Plot
Cov3.plot_error_matrix(Cov3_mat)
"""

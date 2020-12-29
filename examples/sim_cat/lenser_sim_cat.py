import sys,os
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits
import time
from scipy.special import gamma

"""
Module: lenser_sim_cat
.. synopsis: Simulates a catalogue of galaxy images
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. Create a catalogue of simulated galaxy postage stamps using Lenser.
.. A number of desired galaxies Ngal is specified, galactic and lensing 
   parameters are randomly generated for each galaxy, an image is 
   generated for each galaxy, and the resulting Ngal postage stamps are 
   exported to the path 
     'Lenser/examples/Catalogues/Simulated_(insert time)/'
"""

# Create path to catalogue and name the catalogue with the time it was created
tm = time.localtime()
tm_str = str(tm[2])+'_'+str(tm[1])+'_'+str(tm[0])+'_'+str(tm[3])+':'+str(tm[4])+':'+str(tm[5])
path_to_cat = '../Catalogues/'
cat_name = 'Simulated_'+tm_str
cat_folder = cat_name+'/'
image_folder = 'ima/'
os.mkdir(path_to_cat+cat_folder)  
os.mkdir(path_to_cat+cat_folder+image_folder)  

# Number of galaxies in simulated catalogue
Ngal = 1000

# Generate the seeds for np.random. 
# .. We need seeds for the random generation of model parameter, except for the centroid,
# ..  plus I0, which is 13-2+1 seeds, as well as for the random noise generation in each postage stamp.
Npars_seed = 12
seed_list = np.arange(0, Npars_seed+Ngal)

# Choose which galaxy catalogue to mimic: COSMOS, EFIGI, or other.  If other is chosen,
#  you will need to specify the following parameters yourself: 
#  .. Postage stamp size
#  .. Galaxy size in pixels
#  .. Range of galaxy brightness, 
#  .. Stamp noise 
#  .. Sky background
catalogue_type = 'EFIGI'
# COSMOS type catalogue
if catalogue_type == 'COSMOS':
    # Generate non-fit parameters.  
    # .. These values should be motivated to reflect actual data
    # .. Postage stamp size
    Nx = 150
    Ny = 150
    # .. Standard galaxy size (in pixels)
    a = 10.
    # .. I0
    np.random.seed(seed_list[0])
    I0_low, I0_up = 0.1, 10.
    I0_list = np.random.uniform(low=I0_low, high=I0_up, size=Ngal)
    # .. noise1 and noise2
    noise1 = 1.3e-3
    noise2 = 0.
    # .. Background
    background = 0.
    # Comments
    comments = """This catalogue was generated for the purpose of testing the ability of Lenser to reconstruct known parameters.
               The postage stamp size, peak brightness I0, and the noisemaps are all chosen to mimic the COSMOS dataset."""
# EFIGI type catalogue
elif catalogue_type == 'EFIGI':
    # Generate non-fit parameters.  
    # .. These values should be motivated to reflect actual data
    # .. Postage stamp size
    Nx = 255
    Ny = 255
    # .. Standard galaxy size (in pixels)
    a = 25.
    # .. I0
    np.random.seed(seed_list[0])
    I0_low, I0_up = 1.e3, 1.e6
    I0_list = np.random.uniform(low=I0_low, high=I0_up, size=Ngal)
    # .. noise1 and noise2
    noise1 = 2.
    gain = 4.75
    noise2 = 1/np.sqrt(gain)
    # .. Background
    background = 0.
    # Comments
    comments = """This catalogue was generated for the purpose of testing the ability of Lenser to reconstruct known parameters.
               The postage stamp size, peak brightness I0, and the noisemaps (and gains) are all chosen to mimic the EFIGI dataset."""
else:
    print('Other was chosen for catalogue type.  User will specify catalogue-type parameters themselves.')
 
# Randomly generate galaxy parameters 
# .. Set lower and upper bounds for np.random.uniform
# .. Set mean and 1sigma for np.random.normal
# .. Set a seed before each np.random call
# Lensing parameters
# .. First, we have the option whether or not to include shear. By default, shear is included. 
#     The consequence of including shear is that q and phi will not properly be reconstructed by a Lenser fit,
#     due to the shear-ellipticity degeneracy.  It is more realistic, however, to include shear fields if one wishes
#     to use this script in order to test Lenser's ability to fit to itself.
include_shear = True
if include_shear == True:
    # .. We will generate gamma1 and gamma2 from Gaussians and then get psi,ij.
    #     We will set kappa = 0 (we can arbitrarily make this choice due to the 
    #     mass-sheet degeneracy)
    # .. kappa
    kappa = 0.
    # .. shear standard deviation
    sigma_gamma = 0.05
    # .. gamma1
    np.random.seed(seed_list[1])
    gamma1_mean, gamma1_stdv = 0., sigma_gamma/np.sqrt(2)
    gamma1_list = np.random.normal(gamma1_mean, gamma1_stdv, size=Ngal)
    # .. gamma2
    np.random.seed(seed_list[2])
    gamma2_mean, gamma2_stdv = 0., sigma_gamma/np.sqrt(2)
    gamma2_list = np.random.normal(gamma2_mean, gamma2_stdv, size=Ngal)
    # .. psi,11
    psi11_list = kappa + gamma1_list
    # .. psi,12
    psi12_list = gamma2_list
    # .. psi,22
    psi22_list = kappa - gamma1_list
elif include_shear == False:
    # .. psi,11
    psi11 = 0.
    psi11_list = psi11*np.ones(Ngal)
    # .. psi,12
    psi12 = 0.
    psi12_list = psi12*np.ones(Ngal)
    # .. psi,22
    psi22 = 0.
    psi22_list = psi22*np.ones(Ngal)
# .. We have to be careful when generating the flexion, because not all of psi,ijj 
#     are independent from one another. We do the following:
#      (i).   Generate F1 and F2 from Gaussian distributions
#      (ii).  Use F1 and F2 to calculate the angle of flexion, phi_F
#      (iii). Assume a particular analytic lens model, which in this case is a 
#             singular isothermal sphere (SIS).  This allows us to relate first and
#             section flexion in an analytic way.  We then use F1, F2, and phi_F to 
#             get G1 and G2 
#      (iv).  Use F1, F2, G1, and G2 to get psi,ijk
# .. Set sigma_F, the standard deviation for first flexion that is equal to sigma_F1 and
#     sigma_F2 added in quadruture
sigma_F = 1.e-3
# .. F1
np.random.seed(seed_list[3])
F1_mean, F1_stdv = 0., sigma_F/np.sqrt(2)
F1_list = np.random.normal(F1_mean, F1_stdv, size=Ngal)
# .. F2
np.random.seed(seed_list[4])
F2_mean, F2_stdv = 0., sigma_F/np.sqrt(2)
F2_list = np.random.normal(F2_mean, F2_stdv, size=Ngal)
# .. phi_F
# .. .. angle of flexion
phi_F_list = np.arctan2(F2_list,F1_list)
# .. G1
G1_list = -((3*np.cos(3*phi_F_list))/np.cos(phi_F_list))*F1_list
# .. G2
G2_list = -((3*np.sin(3*phi_F_list))/np.sin(phi_F_list))*F2_list
# .. psi,111
psi111_list = (1./2.)*(3.*F1_list + G1_list)
# .. psi,112
psi112_list = (1./2.)*(F2_list + G2_list)
# .. psi,122
psi122_list = (1./2.)*(F1_list - G1_list)
# .. psi,222
psi222_list = (1./2.)*(3.*F2_list - G2_list)
# Shape parameters:
# .. xc:
# .. .. We want the centroid of the galaxy to be able to 
#        float within a single pixel (i.e. dither the centroid)
np.random.seed(seed_list[5])
xc = 0.
xc_list = xc+2.*np.random.random(Ngal)-1
# .. yc
np.random.seed(seed_list[6])
yc = 0.
yc_list = yc+2.*np.random.random(Ngal)-1
# .. ns
# .. .. ns values pulled from a uniform distribution that encompasses most realistic values that 
#        the Sersic model can take
np.random.seed(seed_list[7])
ns_low, ns_up = 0.2, 5#0.1, 6
ns_list = np.random.uniform(low=ns_low, high=ns_up, size=Ngal)
# .. .. with ns generated, we need to adjust the image size
#        a should equal the nominal value for ns leq 1.  But for larger ns, the profile becomes more 
#         disperse and centrally concentrated, and the nominal value for a is too small
id_a1 = np.where((ns_list >= 0.2) & (ns_list <= 1.))
id_a2 = np.where((ns_list > 1.) & (ns_list <= 2.))
id_a3 = np.where((ns_list > 2.) & (ns_list <= 3.))
id_a4 = np.where((ns_list > 3.) & (ns_list <= 4.))
id_a5 = np.where((ns_list > 4.) & (ns_list <= 5.))
a1, a2, a3, a4, a5 = a, a, 2.*a, 4.*a, 8.*a
a_list = np.ones(Ngal)
a_list[id_a1] = a1
a_list[id_a2] = a2
a_list[id_a3] = a3
a_list[id_a4] = a4
a_list[id_a5] = a5
# .. phi
np.random.seed(seed_list[8])
phi_low, phi_up = 0., 2*np.pi
phi_list = np.random.uniform(low=phi_low, high=phi_up, size=Ngal)
# .. q
# .. .. We will make use of the intrisic ellipticity distribution given in Schneider 1996:
np.random.seed(seed_list[9])
sigma_eps = 0.2
# .. .. Generate a list of instrinsic ellipticities twice as long as needed, so we can truncate anywhere where |eps_s| > 0.5
eps_s_list = (1/(np.sqrt(np.pi)*sigma_eps*(1-np.exp(-1/sigma_eps**2.))))*np.random.normal(loc=0,scale=sigma_eps/np.sqrt(2),size=2*Ngal)
id_eps_s = np.where(abs(eps_s_list) < 0.5)
eps_s_list = eps_s_list[id_eps_s]
eps_s_list = eps_s_list[0:Ngal]
# .. .. Get q list
q_list = (1+abs(eps_s_list))/(1-abs(eps_s_list))
# .. .. Now that we have intrinsic ellipticity, get observed q due to eps_s + shear:
eps_s1_list, eps_s2_list = eps_s_list*np.cos(2.*phi_list), eps_s_list*np.sin(2.*phi_list)
eps1_list = eps_s1_list + gamma1_list
eps2_list = eps_s2_list + gamma2_list
eps_list = np.sqrt(eps1_list**2. + eps2_list**2.)
q_obs_list = (1+abs(eps_list))/(1-abs(eps_list))
# .. rs
# .. .. rs is coupled to ns, a, and q.  As such, we should not randomly generate rs values, 
#        rather they should be calculated from the analytic relationship between these parameters 
#        derived by Goldberg and Arena
rs_list = a_list/(np.sqrt(((1+q_obs_list**2.)/2)))*np.sqrt(gamma(2.*ns_list)/gamma(4.*ns_list))


# Create a file to be used as (i) a prep file for lenser_run_cat and 
# (ii) to contain the input shape and lensing parameters for comparison to the fit
prep_params = ['name','class','z','u-r']
input_params = ['Nx', 'Ny', 'a',
                'I0','noise1','noise2','background',
                'xc','yc','ns','rs','q','phi',                                       # Galaxy fit parameters
                'psi11','psi12','psi22','psi111','psi112','psi122','psi222']         # Lensing fit parameters
col_list = prep_params+input_params 
prep_filename = path_to_cat+cat_folder+cat_name+'_info.pkl'
arrs = {k:[] for k in range(len(col_list))} #dict of variable lists

# Generate each postage stamp
for i in range(Ngal):
    xc = xc_list[i]
    yc = yc_list[i]
    ns = ns_list[i]
    rs = rs_list[i]
    q = q_list[i]
    phi = phi_list[i]
    psi11 = psi11_list[i]
    psi12 = psi12_list[i]
    psi22 = psi22_list[i]
    psi111 = psi111_list[i]
    psi112 = psi112_list[i]
    psi122 = psi122_list[i]
    psi222 = psi222_list[i]

    I0 = I0_list[i]
    a = a_list[i]

    myLens = Lens(psi2=[psi11,psi12,psi22],psi3=[psi111,psi112,psi122,psi222]) 
    myGalaxy = Galaxy(xc,yc,ns,rs,q,phi,galaxyLens=myLens) 

    myImage = myGalaxy.generateImage(nx=Nx,ny=Ny,lens=True,I0=I0,
                                     noise1=noise1,noise2=noise2,seed=seed_list[Npars_seed+i],
                                     background=background)

    # Save image to a FITS file
    label=i+1
    hdu=fits.PrimaryHDU(myImage.getMap())
    hdu.writeto(path_to_cat+cat_folder+image_folder+'Galaxy'+str(label)+'.fits',overwrite=True)
    # Save noisemap to a FITS file
    hdu=fits.PrimaryHDU(myImage.getMap(type='noise'))
    hdu.writeto(path_to_cat+cat_folder+image_folder+'Galaxy'+str(label)+'_rms.fits',overwrite=True)

    # Dynamically create prep file
    prep_vals = np.array((['Galaxy'+str(label)],['N/A'],['N/A'],['N/A']))
    input_vals = np.array((Nx, Ny, a,
                           I0,noise1,noise2,background,
                           xc,yc,ns,rs,q,phi,                                       
                           psi11,psi12,psi22,psi111,psi112,psi122,psi222))
    output_vals = np.append(prep_vals, input_vals)

    for i,j in zip(col_list,range(len(col_list))):
        arrs[j].append(output_vals[j])

    # .. Build the dataframe
    dat = {i:arrs[j] for i,j in zip(col_list,range(len(col_list)))}
    out_frame = pd.DataFrame(data=dat,columns=col_list)
    out_frame.to_pickle(prep_filename)

# Save catalogue information to a README
readme = 'short'
# .. Short README file
if readme == 'short':
    readme_arr = np.array((['Created (DD_MM_YYYY_hh:mm:ss):',tm_str],
                           ['Comments:',comments],
                           ['Number of galaxies =',Ngal]))
    np.savetxt(path_to_cat+cat_folder+'README.txt', readme_arr, delimiter=' ', fmt='%s')
# .. Longer README file 
elif readme == 'long':
    if include_shear == True:
        readme_arr = np.array((['Created (DD_MM_YYYY_hh:mm:ss):',tm_str],
                               ['Comments:',comments],
                               ['Number of galaxies =',Ngal],
                               ['Nx =',Nx],
                               ['Ny =',Ny],
                               ['I0 lower bound =',I0_low],
                               ['I0 upper bound =',I0_up],
                               ['noise1 =',noise1],
                               ['noise2 =',noise2],
                               ['background =',background],
                               ['xc =',xc],
                               ['yc =',yc],
                               ['ns lower bound =',ns_low],
                               ['ns upper bound =',ns_up],
                               ['rs lower bound =',rs_low],
                               ['rs upper bound =',rs_up],
                               ['q =','Schneider 1996'],
                               ['phi lower bound =',phi_low],
                               ['phi upper bound =',phi_up],
                               ['psi11 mean =',psi11_mean],
                               ['psi11 stdv =',psi11_stdv],
                               ['psi12 mean =',psi12_mean],
                               ['psi12 stdv =',psi12_stdv],
                               ['psi22 mean =',psi22_mean],
                               ['psi22 stdv =',psi22_stdv],
                               ['psi111 mean =',psi111_mean],
                               ['psi111 stdv =',psi111_stdv],
                               ['psi112 mean =',psi111_mean],
                               ['psi112 stdv =',psi111_stdv],
                               ['psi122 mean =',psi111_mean],
                               ['psi122 stdv =',psi111_stdv],
                               ['psi222 mean =',psi111_mean],
                               ['psi222 stdv =',psi111_stdv]))
    elif include_shear == False:
        readme_arr = np.array((['Created (DD_MM_YYYY_hh:mm:ss):',tm_str],
                               ['Comments:',comments],
                               ['Number of galaxies =',Ngal],
                               ['Nx =',Nx],
                               ['Ny =',Ny],
                               ['I0 lower bound =',I0_low],
                               ['I0 upper bound =',I0_up],
                               ['noise1 =',noise1],
                               ['noise2 =',noise2],
                               ['background =',background],
                               ['xc =',xc],
                               ['yc =',yc],
                               ['ns lower bound =',ns_low],
                               ['ns upper bound =',ns_up],
                               ['rs lower bound =',rs_low],
                               ['rs upper bound =',rs_up],
                               ['q =','Schneider 1996'],
                               ['phi lower bound =',phi_low],
                               ['phi upper bound =',phi_up],
                               ['psi11 =',psi11],
                               ['psi12 =',psi12],
                               ['psi22 =',psi22],
                               ['psi111 mean =',psi111_mean],
                               ['psi111 stdv =',psi111_stdv],
                               ['psi112 mean =',psi111_mean],
                               ['psi112 stdv =',psi111_stdv],
                               ['psi122 mean =',psi111_mean],
                               ['psi122 stdv =',psi111_stdv],
                               ['psi222 mean =',psi111_mean],
                               ['psi222 stdv =',psi111_stdv]))
    np.savetxt(path_to_cat+cat_folder+'README.txt', readme_arr, delimiter=' ', fmt='%s')
else:
   print('Specify readme type.')

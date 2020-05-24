import sys,os
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits
import time

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
tm = time.localtime()
tm_str = str(tm[2])+'_'+str(tm[1])+'_'+str(tm[0])+'_'+str(tm[3])+':'+str(tm[4])+':'+str(tm[5])
path_to_cat = '../Catalogues/'
cat_name = 'Simulated_'+tm_str+'/'
os.mkdir(path_to_cat+cat_name)            

# Number of galaxies in simulated catalogue
Ngal = 1000

# Generate the seends for np.random. 
#- We need seeds for the random generation of model parameter
#- (expect for the centroid)
#- as well as for the random noise generation in each postage stamp
Npars_seed = 8
seed_list = np.arange(0, 8+Ngal)

# Randomly generate galaxy parameter (except centroid and psi,ij)
#- Set lower and upper bounds for np.random.uniform
#- Set mean and 1sigma for np.random.normal
#- Set a seed before each np.random call
#-- xc:
xc = 0.
xc_list = xc*np.ones(Ngal)
#-- yc
yc = 0.
yc_list = yc*np.ones(Ngal)
#-- ns
ns_low, ns_up = 0.5, 1.5
np.random.seed(seed_list[0])
ns_list = np.random.uniform(low=ns_low, high=ns_up, size=(Ngal,))
#-- rs
rs_low, rs_up = 1., 3.
np.random.seed(seed_list[1])
rs_list = np.random.uniform(low=rs_low, high=rs_up, size=(Ngal,))
#-- q
q_low, q_up = 1., 3.5
np.random.seed(seed_list[2])
q_list = np.random.uniform(low=q_low, high=q_up, size=(Ngal,))
#-- phi
phi_low, phi_up = 0., 2*np.pi
np.random.seed(seed_list[3])
phi_list = np.random.uniform(low=phi_low, high=phi_up, size=(Ngal,))
#-- psi,11
psi11 = 0.
psi11_list = psi11*np.ones(Ngal)
#-- psi,12
psi12 = 0.
psi12_list = psi12*np.ones(Ngal)
#-- psi,22
psi22 = 0.
psi22_list = psi22*np.ones(Ngal)
#-- psi,111
psi111_mean, psi111_stdv = 0., 0.001
np.random.seed(seed_list[4])
psi111_list = np.random.normal(psi111_mean, psi111_stdv, size=(Ngal,))
#-- psi,112
psi112_mean, psi112_stdv = 0., 0.001
np.random.seed(seed_list[5])
psi112_list = np.random.normal(psi112_mean, psi112_stdv, size=(Ngal,))
#-- psi,122
psi122_mean, psi122_stdv = 0., 0.001
np.random.seed(seed_list[6])
psi122_list = np.random.normal(psi122_mean, psi122_stdv, size=(Ngal,))
#-- psi,222
psi222_mean, psi222_stdv = 0., 0.001
np.random.seed(seed_list[7])
psi222_list = np.random.normal(psi222_mean, psi222_stdv, size=(Ngal,))


#psi111_list = random.uniform(low=-0.003, high=0.003, size=(Ngal,))
#psi112_list = random.uniform(low=-0.003, high=0.003, size=(Ngal,))
#psi122_list = random.uniform(low=-0.003, high=0.003, size=(Ngal,))
#psi222_list = random.uniform(low=-0.003, high=0.003, size=(Ngal,))

"""
Generate the galaxy images
"""
# Postage stamp size
Nx = 100
Ny = 100
# I0
I0 = 1.e3
# noise1 and noise2
noise1 = 100.
noise2 = 0.
# Background
background = 2.

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

    myLens = Lens(psi2=[psi11,psi12,psi22],psi3=[psi111,psi112,psi122,psi222]) 
    myGalaxy = Galaxy(xc,yc,ns,rs,q,phi,galaxyLens=myLens) 

    myImage = myGalaxy.generateImage(nx=Nx,ny=Ny,lens=True,I0=I0,
                                     noise1=noise1,noise2=noise2,seed=seed_list[Npars_seed+i],
                                     background=background)

    # Save image to a FITS file
    label=i+1
    hdu=fits.PrimaryHDU(myImage.getMap())
    hdu.writeto(path_to_cat+cat_name+'Galaxy'+str(label)+'.fits',overwrite=True)


# Save catalogue information to a README
readme_arr = np.array((['Created (DD_MM_YYYY_hh:mm:ss):',tm_str],
                       ['Number of galaxies =',Ngal],
                       ['Nx =',Nx],
                       ['Ny =',Ny],
                       ['I0 =',I0],
                       ['noise1 =',noise1],
                       ['noise2 =',noise2],
                       ['background =',background],
                       ['xc =',xc],
                       ['yc =',yc],
                       ['ns lower bound =',ns_low],
                       ['ns upper bound =',ns_up],
                       ['rs lower bound =',rs_low],
                       ['rs upper bound =',rs_up],
                       ['q lower bound =',q_low],
                       ['q upper bound =',q_up],
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
np.savetxt(path_to_cat+cat_name+'README.txt', readme_arr, delimiter=' ', fmt='%s')


    



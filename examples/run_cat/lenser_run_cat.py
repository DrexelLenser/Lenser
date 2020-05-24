import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits
import pandas as pd
import glob as glob
import time, sys
from scipy.stats import skew, kurtosis



"""
Module: lenser_run_cat
.. synopsis: Runs an entire catalogue of galaxy images through Lenser and exports pickle file of bestfit parameters
.. module author: Joseph M. Fabritius II <joseph.m.fabritius@drexel.edu>
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. This script will import one of the Catalogues in the Catalougues folder and run all of the images through Lenser.
.. A prep file for a catalogue is required for object identification.
.. Best-fit parameters are dynamically saved in dataframe form to a pickle file.
"""



# Catalogue choice.  One could choose either 'COSMOS' or 'EFIGI'
cat_choice = 'EFIGI'

# If no extra condition is required, use condition = ''
# - You may wish to have condition = _PSF, say, to 
# - differentiate between runs that do or do not 
# - convolve a PSF
# - e.g. condition = '_F814w_COSMOS-noisemaps_no-PSF'
condition = ''

def get_image_data(name, path):
    """
    Image data grabber
    """
    fit_file = fits.open(glob.glob(path+'*'+name+'*.fits')[0])
    return fit_file[0].data 

def update_progress(job_title, progress):
    """
    Progress bar
    """
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: |{1}| {2}%".format(job_title, "[]"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " Complete\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

# Prepare Catalogue
if cat_choice == 'EFIGI':
    path_to_cat = '../Catalogues/EFIGI/'
    prep_file = 'EFIGI_catalog_w_color.pkl'
    im_fold = 'ima_r/'
    input_params = ['name','class','z','u-r']
elif cat_choice == 'COSMOS':
    path_to_cat = '../Catalogues/COSMOS/'
    prep_file = 'COSMOS-morph-info-rev1.pkl'
    im_fold = 'ima_r/'
    input_params = ['name','class','z','U-R']
else:
    print('Error! Need a catalog choice')

# Parameters for pickle file 
output_params = ['x_c','y_c','ns','r_s','q','phi',                                    # Galaxy fit parameters
                 'psi_11','psi_12','psi_22','psi_111','psi_112','psi_122','psi_222',  # Lensing fit parameters
                 'rchi2',                                                             # chisquared of fit
                 'F1_fit', 'F2_fit',                                                  # Flexion from fit                                       
                 'a','Q111','Q112','Q122','Q222',                                     # Image size and octupole moments
                 'F1_HOLICs', 'F2_HOLICs',                                            # Flexion from HOLICs
                 'mask_flag',                                                         # Mask flag: =1 if mask extends outside stamp
                 'checkFit_flag',                                                     # checkFit flag: =1 if multiple images appear within stamp
                 'noisemap_masked_mean', 'noisemap_masked_stdv',                      # Noisemap statistics
                 'noisemap_masked_skew','noisemap_masked_kurtosis']                   # --
col_list = input_params+output_params 
output_filename = cat_choice+'_'+str(time.localtime()[1])+str(time.localtime()[2])+str(time.localtime()[0])
arrs = {k:[] for k in range(len(input_params+output_params))} #dict of variable lists

# Get the list of objects to loop through LENSER
prep_frame = pd.read_pickle(path_to_cat+prep_file)
prep_frame=prep_frame[(prep_frame['class'] != 'irregular')]
ID_list = prep_frame.index.values #list of index values

# Lenser Analysis Loop
init = 0
for _ in ID_list:
    # get the name of object with index _
    imname = prep_frame.name.loc[_] 
    print(' \n')
    print('Object '+str(init+1)+' - '+imname)
    init+=1
    
    try:
        # Lenser analysis
        print('Running Lenser')

        # .. Get image
        myImage = Image(path_to_cat+im_fold+imname+'_r.fits')

        # .. Generate mask
        myImage.generateMask(subtractBackground=True)
        
        # .. Mask flag: =1 if mask extends outside stamp
        mask_flag_val = 0
        mask = myImage.maskmap
        for i in np.arange(mask.shape[0]):
            if (mask[0][i] != 0) or (mask[i][0] != 0) or (mask[i][-1] != 0) or (mask[-1][i] != 0):
                mask_flag_val = 1
                break
                
        # .. Initialize AIM model
        myModel = aimModel(myImage)

        # .. Run local minimization
        myModel.runLocalMinRoutine()

        # .. Check fit
        checkFit_flag_val = 0
        checkFit_flag_val = myModel.checkFit()
        
        # .. Get flexion from fit
        F, G = myModel.psi3ToFlexion()
        F1_fit = F[0]
        F2_fit = F[1]

        # .. Get size, octupole moments
        order2,order3,order4 = myImage.getMoments('order2,order3,order4')
        quadrupole = order2
        octupole = order3
        Q11 = quadrupole[0]
        Q12 = quadrupole[1]
        Q22 = quadrupole[2]
        Q111 = octupole[0]
        Q112 = octupole[1]
        Q122 = octupole[2]
        Q222 = octupole[3]
        a = np.sqrt(abs(Q11+Q22))

        # .. Check to see if any image moments are NaNs:
        # .. if they are, then terminate loop
        if np.isnan(a) or np.isnan(Q111) or np.isnan(Q112) or np.isnan(Q122) or np.isnan(Q222):
            continue

        # .. Get flexion from HOLICs
        F_HOLICs, G_HOLICs, = myModel.flexionMoments()
        F1_HOLICs = F_HOLICs[0]
        F2_HOLICs = F_HOLICs[1]

        # .. Get mean, stdv, skew, and kurtosis of masked noisemap
        myNoise_masked = myImage.getMap(type='noise_masked')
        noisemap_masked_mean = np.mean(myNoise_masked.flatten())
        noisemap_masked_stdv = np.std(myNoise_masked.flatten())
        noisemap_masked_skew = skew(myNoise_masked.flatten())
        noisemap_masked_kurtosis = kurtosis(myNoise_masked.flatten())   

        # .. Get values to save
        fit_pars = myModel.parsWrapper()
        other_pars = np.array((myModel.chisq(),
                               F1_fit, F2_fit,
                               a, Q111, Q112, Q122, Q222,
                               F1_HOLICs, F2_HOLICs,
                               mask_flag_val,
                               checkFit_flag_val,
                               noisemap_masked_mean, noisemap_masked_stdv,
                               noisemap_masked_skew, noisemap_masked_kurtosis))
        output_vals = np.append(fit_pars, other_pars)

        # .. Empty model
        myModel.empty()

        # Export values to pickle file

        if cat_choice=='EFIGI':imname=imname.split('_')[0]

        # .. Build dictionary for quick, dynamic dataframe building per iteration
        for i,j in zip(col_list,range(len(col_list))):
            if i in output_params: #if output parameter, append value appropriately
                arrs[j].append(output_vals[abs(len(input_params)-j)])
            else:
                sel_id = prep_frame[prep_frame.name==imname].index[0]
                prep_val = prep_frame[i].loc[sel_id]
                arrs[j].append(prep_val)
                
        # .. Build the dataframe
        dat = {i:arrs[j] for i,j in zip(col_list,range(len(col_list)))}
        out_frame = pd.DataFrame(data=dat,columns=col_list)
        out_frame.to_pickle(path_to_cat+output_filename+'-'+str(len(ID_list))+'_objects'+condition+'.pkl')
        
    except:
        print('Error, skipping')

    # Print progress bar
    update_progress('Lenser Analysis - '+cat_choice, init/float(len(ID_list)))


import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits
import pandas as pd
import time, sys



"""
Module: lenser_run_cat_multi_fit
.. synopsis: Runs an entire catalogue of galaxy images through Lenser in multi-fit mode 
             and exports pickle file of bestfit parameters
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. This script will import one of the Catalogues in the Catalougues folder and run all of the images through Lenser.
.. A prep file for a catalogue is required for object identification.
.. Best-fit parameters are dynamically saved in dataframe form to a pickle file.
"""

# Catalogue choice
cat_choice = 'COSMOS'
# Choice of bands
bands = ['F814W', 'F606W', 'F125W']
# If no extra condition is required, use condition = ''
# .. You may wish to have condition = _PSF, say, to 
# .. differentiate between runs that do or do not 
# .. convolve a PSF
# .. e.g. condition = '_F814W_COSMOS_no-PSF'
condition = ''
    
# Prepare Catalogue
if cat_choice == 'COSMOS':
    path_to_cat = '../Catalogues/COSMOS/'
    prep_file = 'COSMOS-morph-info-rev1.pkl'
    im_fold = ['Images_'+bands[i]+'/' for i in range(len(bands))]
    input_params = ['name','class','z','U-R']
else:
    print('Error! Need a catalog choice.')

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

# Parameters for pickle file 
output_params = ['xc','yc','ns','rs','q','phi',                                       # Galaxy fit parameters
                 'psi11','psi12','psi22','psi111','psi112','psi122','psi222',         # Lensing fit parameters
                 'err_xc','err_yc','err_ns','err_rs','err_q','err_phi',               # Error on galaxy fit parameters
                 'err_psi11','err_psi12','err_psi22',                                 # Error on lensing fit parameters
                 'err_psi111','err_psi112','err_psi122','err_psi222',                 # --
                 'rchi2',                                                             # chisquared of fit
                 'F1_fit', 'F2_fit',                                                  # Flexion from fi
                 'G1_fit', 'G2_fit',                                                  # --
                 'I0', 'a']                                                           # Galaxy normalization and size

# Prepare pickle file
col_list = input_params+output_params 
output_filename = cat_choice+'_'+str(time.localtime()[1])+str(time.localtime()[2])+str(time.localtime()[0])
# .. dict of variable lists
arrs = {k:[] for k in range(len(input_params+output_params))} 

# Get the list of objects to loop through LENSER
# .. Read in prep frame file
prep_frame = pd.read_pickle(path_to_cat+prep_file)
# .. Optional: do not run galaxies of irregular type through Lenser
#prep_frame = prep_frame[(prep_frame['class'] != 'irregular')]
# .. Get ID list
ID_list = prep_frame.index.values 

# Lenser Analysis Loop
init=0
for i in ID_list:
    # get the name of object with index i
    objname = prep_frame.name.loc[i] 
    print(' \n')
    print('Object '+str(i)+' - '+objname)
    init+=1
    
    try:
        # Lenser analysis
        print('Running Lenser')

        # .. Prepare lists for multi-band fitting
        name_list = []
        dat_list = []
        rms_list = []
        seg_list = []
        psf_list = []
        bg_list = []
        
        for j in range(len(bands)):
            # .. Image name
            imname = objname+'_'+bands[j]
            # .. Read in image from FITS file
            path_to_image = path_to_cat+im_fold[j]+imname+'.fits'
            f = FITS(path_to_image)
            dat = f.get_FITS('data')
            rms = f.get_FITS('noise')
            seg = f.get_FITS('segmask')
            psf = f.get_FITS('psf')
            bg = f.get_FITS('bgmask')
            # .. Append multi-band list
            name_list.append(imname)
            dat_list.append(dat)
            rms_list.append(rms)
            seg_list.append(seg)
            psf_list.append(psf)
            bg_list.append(bg)

        # .. Now do multi-fit
        multiname = objname+'_multi_band_fit'
        myMultiImage = MultiImage(namelist = name_list, datalist = dat_list, noiselist = rms_list,
                                  seglist = seg_list, psflist = psf_list, bgmasklist = bg_list)
        # .. Initialize AIM model
        myModel = aimModel(myMultiImage = myMultiImage)
        # Run local minimization
        myModel.runLocalMinRoutine()
        # Get flexion from fit
        F, G = myModel.psi3ToFlexion()
        F1_fit = F[0]
        F2_fit = F[1]
        G1_fit = G[0]
        G2_fit = G[1]
        # Get size
        a = myModel.size() 
        # Get values to save
        fit_pars = np.append(myModel.parsWrapper(), myModel.parsErrorWrapper())
        other_pars = np.array((myModel.chisquared,
                               F1_fit, F2_fit,
                               G1_fit, G2_fit,
                               myModel.I0, a))
        output_vals = np.append(fit_pars, other_pars)
        # Empty model
        myModel.empty()

    except:# ValueError:
        print('Error, skipping')
        output_vals = np.nan*np.ones(len(output_params))

    # .. Build dictionary for quick, dynamic dataframe building per iteration
    for k,l in zip(col_list,range(len(col_list))):
        if k in output_params: #if output parameter, append value appropriately
            arrs[l].append(output_vals[abs(len(input_params)-l)])
        else:
            sel_id = prep_frame[prep_frame.name==objname].index[0]
            prep_val = prep_frame[k].loc[sel_id]
            arrs[l].append(prep_val)
    # .. Build the dataframe
    data = {k:arrs[l] for k,l in zip(col_list,range(len(col_list)))}
    out_frame = pd.DataFrame(data=data,columns=col_list)
    out_frame.to_pickle(path_to_cat+output_filename+'-'+str(len(ID_list))+'_objects_multi-fit'+condition+'.pkl')

import sys
sys.path.append('../..')
from lenser import *
import numpy as np
from astropy.io import fits
import pandas as pd
import time, sys



"""
Module: lenser_run_cat_single_fit
.. synopsis: Runs an entire catalogue of galaxy images through Lenser in single-fit mode 
             and exports pickle file of bestfit parameters
.. module author: Joseph M. Fabritius II <joseph.m.fabritius@drexel.edu>
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>

.. This script will import one of the Catalogues in the Catalougues folder and run all of the images through Lenser.
.. A prep file for a catalogue is required for object identification.
.. Best-fit parameters are dynamically saved in dataframe form to a pickle file.
"""

# Catalogue choice.  One could choose either 'COSMOS' or 'EFIGI'
cat_choice = 'COSMOS'
# Band choice. Choose from Hubble filters for COSMOS or SDSS filters for EFIGI
# .. one can also choose no band with = ''
band = 'F814W'
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
    im_fold = 'Images_'+band+'/'
    input_params = ['name','class','z','U-R']
elif cat_choice == 'EFIGI':
    path_to_cat = '../Catalogues/EFIGI/'
    prep_file = 'EFIGI_catalog_w_color.pkl'
    im_fold = 'Images_'+band+'/'
    input_params = ['name','class','z','u-r']
elif 'Simulated' in cat_choice:
    path_to_cat = '../Catalogues/'+cat_choice+'/'
    prep_file = cat_choice+'_info.pkl'
    im_fold = 'Images/'
    input_params = ['name','class','z','u-r']
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
                 'F1_HOLICs', 'F2_HOLICs',                                            # Flexion from HOLICs
                 'G1_HOLICs', 'G2_HOLICs',                                            # --
                 'I0', 'a',                                                           # Galaxy normalization and size
                 'mask_flag']                                                         # Mask flag: =1 if mask extends outside stamp

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

        # .. Image name
        imname = objname+'_'+band

        # .. Read in image from FITS file
        path_to_image = path_to_cat+im_fold+imname+'.fits'
        f = FITS(path_to_image)
        dat = f.get_FITS('data')
        rms = f.get_FITS('noise')
        seg = f.get_FITS('segmask')
        psf = f.get_FITS('psf') 
        bg = f.get_FITS('bgmask')
        
        # .. Get image
        myImage = Image(name = imname, datamap = dat,
                        noisemap = rms, segmask = seg, psfmap = psf, bgmask = bg)

        # .. Generate mask
        myImage.generateEllipticalMask(subtractBackground=True)
        
        # .. Mask flag: =1 if mask extends outside stamp
        mask_flag_val = 0
        mask = myImage.getMap('totalmask')
        for m in np.arange(mask.shape[0]):
            if (mask[0][m] != 0) or (mask[m][0] != 0) or (mask[m][-1] != 0) or (mask[-1][m] != 0):
                mask_flag_val = 1
                break
                
        # .. Initialize AIM model
        myModel = aimModel(myImage)

        # .. Run local minimization
        myModel.runLocalMinRoutine()
        
        # .. Get flexion from fit
        F, G = myModel.psi3ToFlexion()
        F1_fit = F[0]
        F2_fit = F[1]
        G1_fit = G[0]
        G2_fit = G[1]
        
        # .. Get size, octupole moments
        a = myModel.size()
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

        # .. Check to see if any image moments are NaNs:
        # .. if they are, then terminate loop
        if np.isnan(a) or np.isnan(Q111) or np.isnan(Q112) or np.isnan(Q122) or np.isnan(Q222):
            # .. Empty model
            myModel.empty()
            raise ValueError
            #continue

        # .. Get flexion from HOLICs
        F_HOLICs, G_HOLICs = myModel.flexionMoments(myImage)
        F1_HOLICs = F_HOLICs[0]
        F2_HOLICs = F_HOLICs[1]
        G1_HOLICs = G_HOLICs[0]
        G2_HOLICs = G_HOLICs[1]

        # .. Get values to save
        fit_pars = np.append(myModel.parsWrapper(), myModel.parsErrorWrapper())
        other_pars = np.array((myModel.chisquared,
                               F1_fit, F2_fit,
                               G1_fit, G2_fit,
                               F1_HOLICs, F2_HOLICs,
                               G1_HOLICs, G2_HOLICs,
                               myModel.I0, a,
                               mask_flag_val))
        output_vals = np.append(fit_pars, other_pars)

        # .. Empty model
        myModel.empty()
        
    except:# ValueError:
        print('Error, skipping')
        output_vals = np.nan*np.ones(len(output_params))

    # Export values to pickle file

    if cat_choice=='EFIGI':imname=imname.split('_')[0]

    # .. Build dictionary for quick, dynamic dataframe building per iteration
    for k,l in zip(col_list,range(len(col_list))):
        if k in output_params: #if output parameter, append value appropriately
            arrs[l].append(output_vals[abs(len(input_params)-l)])
        else:
            sel_id = prep_frame[prep_frame.name==objname].index[0]
            prep_val = prep_frame[k].loc[sel_id]
            arrs[l].append(prep_val)
    # .. Build the dataframe
    dat = {k:arrs[l] for k,l in zip(col_list,range(len(col_list)))}
    out_frame = pd.DataFrame(data=dat,columns=col_list)
    out_frame.to_pickle(path_to_cat+output_filename+'-'+str(len(ID_list))+'_objects_single-fit'+condition+'.pkl')

    # Print progress bar
    update_progress('Lenser Analysis - '+cat_choice, (init)/float(len(ID_list)))

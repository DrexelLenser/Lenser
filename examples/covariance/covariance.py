"""
Module: covariance
.. synopsis: Calculates a covariance matrix for the Lenser parameter space
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>
"""

import sys,os
sys.path.append('../..')
from lenser import *
from astropy.io import fits
import numpy as np
import pickle
import scipy
import pandas as pd
import glob
import time
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

#To use LaTeX and select Helvetica as the default font:
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

class Covariance(object):

    """
    Covariance class:
    .. Since Lenser is a forward-modeling code, the user can specify a set of input parameters and 
       create an image of a lensed galaxy. It is, therefore, possible to use Lenser in order to 
       compute a covariance matrix for our parameter space by simulating an ensemble of postage 
       stamp images (a "stamp collection") with known input parameters and noise, and 
       then running each of the postage stamps through Lenser for fitting. To test the response of 
       Lenser to noise, each postage stamp has identical input parameters and noise maps, but 
       additional, unique Gaussian noise injected into each.
    .. This module first creates a stamp collection, computes the covariance matrix, and then creates
       a triangle plot of 1- and 2-sigma error ellipses, along with the fiducial parameters indicated
       by a white plus sign.
    """
    
    def __init__(self, Nx=100, Ny=100,
                        xc=0, yc=0, ns=0.75, rs=2., q=3.5, phi=np.pi/6,
                        psi2=[0,0,0], psi3=[0,0,0,0],
                        marg=np.ones(13),
                        I0=1.e3, noise1=1.e3/1.e4, noise2=0, background=0,
                        N_iter = 20,
                        fid_params=None,
                        stamp_col_label='',
                        path_to_col=None,pickle_file=None):
        self.Nx=Nx
        self.Ny=Ny
        self.xc=xc
        self.yc=yc
        self.ns=ns
        self.rs=rs
        self.q=q
        self.phi=phi
        self.psi2=psi2
        self.psi3=psi3
        self.marg=marg
        self.I0=I0
        self.noise1=noise1
        self.noise2=noise2
        self.background=background
        self.N_iter=N_iter
        self.fid_params=fid_params
        self.stamp_col_label=stamp_col_label
        self.path_to_col=path_to_col
        self.pickle_file=pickle_file
        self.covariance_matrix=None
        self.cms=[]
        self.Ncms=len(self.cms)

    def params(self):
        """Modified Sersic model input parameter values
        """
        pars = np.array((self.xc,
                         self.yc,
                         self.ns,
                         self.rs,
                         self.q,
                         self.phi,
                         self.psi2[0],
                         self.psi2[1],
                         self.psi2[2],
                         self.psi3[0],
                         self.psi3[1],
                         self.psi3[2],
                         self.psi3[3]))
        pars = pars[np.where(self.marg==1)]
        self.N_pars = len(pars)
        return pars
            
    def fiducialParams(self):
        """Fiducial parameter values i.e. the values nominally reconstructed by Lenser.  
           These will differ from actual input parameters in the presence of nonzero shear 
           i.e. q and phi change when shear is introduced.  If no shear is present, then
           the fiducial parameters should simply be taken to be the input parameters, 
           appropriately marginalized
        """
        if self.fid_params is not None:
            return self.fid_params
        elif self.fid_params is None:
            return self.params()

    def paramLaTeXNames(self):
        par_names =  [r'$\theta_0^1$',
                      r'$\theta_0^2$',
                      r'$n_s$',
                      r'$\theta_s$',
                      r'$q$',
                      r'$\phi$',
                      r'$\psi,_{11}$',
                      r'$\psi,_{12}$',
                      r'$\psi,_{22}$',
                      r'$\psi,_{111}$',
                      r'$\psi,_{112}$',
                      r'$\psi,_{122}$',
                      r'$\psi,_{222}$']
        par_names = np.array(par_names)
        par_names = par_names[np.where(self.marg==1)]
        return par_names

    def paramNames(self):
        par_names =  ['xc',
                      'yc',
                      'ns',
                      'rs',
                      'q',
                      'phi',
                      'psi11',
                      'psi12',
                      'psi22',
                      'psi111',
                      'psi112',
                      'psi122',
                      'psi222']
        par_names = np.array(par_names)
        par_names = par_names[np.where(self.marg==1)]
        return par_names


    def simulateGals(self, overwrite=False):
        """Simulate a stamp collection
        """
        # Create a directory for the stamp collections, if one does not exist
        if not os.path.exists('stamp_collections/'):
            os.mkdir('stamp_collections/')

        # Make a directory for the stamp collection of choice
        tm = time.localtime()
        tm_str = str(tm[2])+'_'+str(tm[1])+'_'+str(tm[0])+'_'+str(tm[3])+':'+str(tm[4])+':'+str(tm[5])
        if self.stamp_col_label != '':
            stamp_col_name = 'stamp_collection_'+self.stamp_col_label+'/'
        else:
            stamp_col_name = 'stamp_collection/'

        # Make directory for catalogue
        self.path_to_col = 'stamp_collections/'+stamp_col_name
        
        # If this directory already exists (i.e. this particular catalogue has already been simulated), 
        #  then do not simulate it again unless overwrite = True.
        if not os.path.exists(self.path_to_col) or overwrite == True:
            if not os.path.exists(self.path_to_col):
                os.mkdir(self.path_to_col)
                os.mkdir(self.path_to_col+'stamps/')

            # Create README with list of parameters used in simulation
            readme_arr = np.array((['Created (DD_MM_YYYY_hh:mm:ss):',tm_str],
                                   ['Nx =',self.Nx],
                                   ['Ny =',self.Ny],
                                   ['xc =',self.xc],
                                   ['yc =',self.yc],
                                   ['ns =',self.ns],
                                   ['rs =',self.rs],
                                   ['q =',self.q],
                                   ['phi =',self.phi],
                                   ['psi2 =',self.psi2],
                                   ['psi3 =',self.psi3],
                                   ['I0 =',self.I0],
                                   ['noise1 =',self.noise1],
                                   ['noise2 =',self.noise2],
                                   ['background =',self.background],
                                   ['N_iter =',self.N_iter]))

            np.savetxt(self.path_to_col+'README.txt', readme_arr, delimiter=' ', fmt='%s')

            # Create the seeds for numpy.random.seed
            seeds = np.arange(0,self.N_iter)

            # Create lens
            myLens = Lens(psi2=self.psi2,psi3=self.psi3)

            # We want the centroid of the galaxy to be able to 
            #  float within a single pixel for each of the n
            #  iterations (i.e. we are dithering the centroid)
            xc_list = []
            yc_list = []
            for i in range(self.N_iter):
                np.random.seed(seeds[i])
                xc = self.xc+np.random.random(1)/2.
                xc_list.append(xc)
                np.random.seed(seeds[i]+1)
                yc = self.yc+np.random.random(1)/2.
                yc_list.append(yc)

            # Create the n different galaxy instances
            #  (all galaxy instaces are roughly the same galaxy but with
            #   a dithered centroid)
            myGalaxy_list = []
            for i in range(self.N_iter):
                myGalaxy=Galaxy(xc=xc_list[i], yc=yc_list[i],
                                ns=self.ns, rs=self.rs, q=self.q, phi=self.phi,
                                galaxyLens=myLens) 
                myGalaxy_list.append(myGalaxy)

            # Generate the different images with (roughly) the same
            #  galaxy but varying datamaps due to randomly generated noise

            # .. First let us generate the n different images
            myImage_list = []
            for i in range(self.N_iter):
                myImage = myGalaxy_list[i].generateImage(self.Nx,self.Ny,
                                                         lens=True,
                                                         I0=self.I0,noise1=self.noise1,noise2=self.noise2,seed=seeds[i],background=self.background)
                myImage_list.append(myImage)

            # .. Now, let's save the datamaps and noisemaps to FITS files
            for i in range(self.N_iter):
                hdu_datamap=fits.PrimaryHDU(myImage_list[i].getMap(type='data'))
                hdu_datamap.writeto(self.path_to_col+'stamps/'+'Simulated_Galaxy_'+str(i)+'.fits',clobber=True)

            for i in range(self.N_iter):
                hdu_datamap=fits.PrimaryHDU(myImage_list[i].getMap(type='noise'))
                hdu_datamap.writeto(self.path_to_col+'stamps/'+'Simulated_Galaxy_'+str(i)+'_rms.fits',clobber=True)


    def lenserRun(self, overwrite=False):
        """Run lenser on the stamp catalogue
        """

        gal_list = []
        for fits_file in glob.glob(self.path_to_col+'stamps/'+'/*.fits'):
            if 'rms' not in fits_file:
                gal_list.append(fits_file)
            else:
                continue

        Ngals = len(gal_list)
        # Parameters for pickle file 
        output_params = ['xc','yc','ns','rs','q','phi',                                       # Galaxy fit parameters
                         'psi11','psi12','psi22','psi111','psi112','psi122','psi222',         # Lensing fit parameters
                         'I0',                                                                # I0
                         'rchi2',                                                             # chisquared of fit
                         'F1_fit', 'F2_fit',                                                  # Flexion from fit                                       
                         'a',                                                                 # Image size
                         'F1_HOLICs', 'F2_HOLICs',                                            # Flexion from HOLICs
                         'mask_flag',                                                         # Mask flag: =1 if mask extends outside stamp
                         'checkFit_flag',                                                     # checkFit flag: =1 if multiple images appear within stamp
                         'noisemap_masked_mean', 'noisemap_masked_stdv',                      # Noisemap statistics
                         'noisemap_masked_skew','noisemap_masked_kurtosis']                   # --
        output_filename = 'parameter_bestfits'
        self.pickle_file = self.path_to_col+output_filename+'.pkl'
        arrs = {k:[] for k in range(len(output_params))} #dict of variable lists
        col_list = output_params

        # If Lenser has already been run on this catalogue, 
        #  then do not run it again unless overwrite = True.
        if not os.path.exists(self.pickle_file) or overwrite == True:
        
            for i in range(Ngals):

                name = str(gal_list[i].split(self.path_to_col+'/')[-1].split('.fits')[0])
                print('Input galaxy: ', name)

                try:

                    # Read in image from FITS file
                    myImage=Image(gal_list[i])

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

                    # .. Get size
                    a = myModel.size()

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
                    other_pars = np.array((myModel.I0,
                                           myModel.chisq(),
                                           F1_fit, F2_fit,
                                           a, 
                                           F1_HOLICs, F2_HOLICs,
                                           mask_flag_val,
                                           checkFit_flag_val,
                                           noisemap_masked_mean, noisemap_masked_stdv,
                                           noisemap_masked_skew, noisemap_masked_kurtosis))
                    output_vals = np.append(fit_pars, other_pars)

                    # Reset the parameters to their default values
                    # This step is needed if you are looping over multiple images
                    myModel.empty()

                    #build dictionary for quick, dynamic dataframe building per iteration
                    for i,j in zip(col_list,range(len(col_list))):
                        if i in output_params: #if output parameter, append value appropriately
                            arrs[j].append(output_vals[j])

                    #build the dataframe
                    dat = {i:arrs[j] for i,j in zip(col_list,range(len(col_list)))}
                    out_frame = pd.DataFrame(data=dat,columns=col_list)
                    out_frame.index.name=name
                    out_frame.to_pickle(self.pickle_file)

                except:
                    print('Error in fit, skipping')

    #def paramMeans(self):

    def computeCovMat(self):

        # Import tables from pickle files
        pickle_parameters = glob.glob(self.pickle_file)[0]
        params = pickle.load(open(pickle_parameters, 'rb'),encoding='latin1')

        # Get chisqr and set id to where the fit is successful (no NaNs) and < 1.5
        chisq = params['rchi2'].to_numpy()
        checkFit_flag = params['checkFit_flag'].to_numpy()
        mask_flag = params['mask_flag'].to_numpy()
        a = params['a'].to_numpy()
        F = np.sqrt(params['F1_fit'].to_numpy()**2. + params['F2_fit'].to_numpy()**2.)
        id = np.where((~np.isnan(chisq)) & (chisq < 1.5) & (a > 4.5) & (a*F < 1) & (checkFit_flag ==0) & (mask_flag==0))

        # Get lists of individual parameters
        xc_list = params['xc'].to_numpy()[id]
        yc_list = params['yc'].to_numpy()[id]
        ns_list = params['ns'].to_numpy()[id]
        rs_list = params['rs'].to_numpy()[id]
        q_list = params['q'].to_numpy()[id]
        phi_list = params['phi'].to_numpy()[id]
        psi11_list = params['psi11'].to_numpy()[id]
        psi12_list = params['psi12'].to_numpy()[id]
        psi22_list = params['psi22'].to_numpy()[id]
        psi111_list = params['psi111'].to_numpy()[id]
        psi112_list = params['psi112'].to_numpy()[id]
        psi122_list = params['psi122'].to_numpy()[id]
        psi222_list = params['psi222'].to_numpy()[id]

        # Create parameter matrix
        pmat = np.vstack((xc_list,
                          yc_list,
                          ns_list,
                          rs_list,
                          q_list,
                          phi_list,
                          psi11_list,
                          psi12_list,
                          psi22_list,
                          psi111_list,
                          psi112_list,
                          psi122_list,
                          psi222_list))
        
        # Marginalize where needed
        pmat = pmat[np.where(self.marg==1)]
        self.pmat = pmat

        # Get number of parmeters
        N_pars = len(pmat[:,0])
        self.N_pars=N_pars

        # Get number of iterations
        N_iter = len(pmat[0])

        # Calculate covariance matrix
        cov = np.zeros((N_pars, N_pars))
        for i in range(N_pars):
            for j in range(N_pars):
                for k in range(N_iter):
                    cov[i][j] += (pmat[i][k]-np.mean(pmat[i]))*(pmat[j][k]-np.mean(pmat[j]))/(N_iter-1)
        
        # Save covariance matrix
        np.savetxt(self.path_to_col+'covariance_matrix.dat', cov, delimiter=' & ')

        self.covariance_matrix=cov

        return cov

    def covariance_array(self, *covariance_matrices):
        """Creates an array of covariance matrices for combined visualization
        This assumes that for both covariance matrices, the modified Sersic 
        model fiducial parameters are identical. 
        """
        self.cms=covariance_matrices
        self.Ncms=len(covariance_matrices)
        return self.cms

    def error(self, covariance_matrix):
        """Computes the marginalized 1-sigma uncertainty on each parameter
        """
        errors=[]
        for i in range(self.N_pars):
            err=np.sqrt(covariance_matrix[i,i])
            errors.append(err)
        return errors

    def marginalize(self, covariance_matrix, i, j):
        """Compute and return a new covariance matrix after marginalizing over all
        other parameters not in the list param_list
        """
        param_list=[i,j]

        mlist=[]
        for k in range(self.N_pars):
            if k not in param_list:
                mlist.append(k)
        
        trun_cov_matrix=covariance_matrix        
        trun_cov_matrix=np.delete(trun_cov_matrix, mlist, 0)
        trun_cov_matrix=np.delete(trun_cov_matrix, mlist, 1)
        return trun_cov_matrix

    def error_ellipse(self, covariance_matrix, i, j, nstd=1, space_factor=5., clr='k', alpha=0.5, lw=1.5, zorder=0, color=False):
        """Compute 2D marginalized confidence ellipses
        """
        if color==True:
            clr='b'
        def eigsorted(cov):
            vals, vecs=np.linalg.eigh(cov)
            order=vals.argsort()[::1]
            return vals[order], vecs[:,order]

        marginal_covariance_matrix=self.marginalize(covariance_matrix, i,j)
        vals, vecs=eigsorted(marginal_covariance_matrix)
        theta=np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height=2*nstd*np.sqrt(np.absolute(vals))
        xypos=[np.mean(self.pmat[i]), np.mean(self.pmat[j])]
        ellip=Ellipse(xy=xypos, width=width, height=height, angle=theta, color=clr, alpha=alpha, zorder=zorder)
        ellip.set_linewidth(lw)
        ellip_vertices=ellip.get_verts()
        xl=[ellip_vertices[k][0] for k in range(len(ellip_vertices))]
        yl=[ellip_vertices[k][1] for k in range(len(ellip_vertices))]
        dx=(max(xl)-min(xl))/space_factor
        dy=(max(yl)-min(yl))/space_factor
        xyaxes=[min(xl)-dx, max(xl)+dx, min(yl)-dy, max(yl)+dy]

        return ellip, xyaxes

    def oneD_constraints(self, covariance_matrix, i, j, nstd=1., space_factor=5., clr='k', lw=1.5, color=False):
        """Compute 1D marginalized constraints
        """
        if color==True:
            clr='b'
        # Ensure that we are computing the diagonal of the triangle plot
        assert(i==j)
         
        marginal_covariance_matrix=self.marginalize(covariance_matrix, i, j)
        xpos=np.mean(self.pmat[i])

        sig=np.sqrt(np.absolute(marginal_covariance_matrix))
        sig=sig.item()
        xx=np.linspace(xpos-20.*sig, xpos+20.*sig, 4000)
        yy=1./np.sqrt(2.*np.pi*sig**2.) * np.exp(-0.5 * ((xx-xpos)/sig)**2.)
        yy/=np.max(yy)

        dx=((max(xx)-min(xx)))/space_factor
        dy=(max(yy)-min(yy))/space_factor
        xyaxes=[(min(xx)-dx)/10., (max(xx)+dx)/10., min(yy)-dy, max(yy)+dy]

        return yy, xx, xyaxes

    def plot_error_ellipse(self, covariance_matrix, i, j, xyaxes_input=0, nstd=1, clr='k', alpha=0.5, lw=1.5, zorder=0, color=False):
        """Plot the 2D marginalized error ellipses
        """
        if color==True:
            clr='b'
        ax = plt.gca()
        errorellipse, xyaxes=self.error_ellipse(covariance_matrix, i, j, nstd=nstd, clr=clr, alpha=alpha, lw=lw, zorder=zorder)
        ax.add_artist(errorellipse)
        if (xyaxes_input!=0):
            ax.axis(xyaxes_input)
        else:
            ax.axis(xyaxes)

    def plot_oneD_constraints(self, covariance_matrix, i, j, clr='k', lw=1.5, color=False):
        """Plot the 1D marginalized Gaussians
        """
        if color==True:
            clr='b'
        ax=plt.subplot(111)
        y,x=self.oneD_constraints(covariance_matrix, i, j, clr=clr, lw=lw)
        ax.plot(x,y)
        plt.show()

    def plot_error_matrix(self, covariance_matrix, figname=None, nstd=1, nbinsx=3, nbinsy=3, color=False):
        """Create a triangle plot of 2D error ellipses and 1D Gaussians
        given the list of parameters provided
        """
        if color==True:
            clr='cornflowerblue'
        else:
            clr='k'
        Np=self.N_pars

        f, allaxes = plt.subplots(Np, Np, sharex="col", figsize=(Np,Np))#, sharey="row")

        for j in range(Np):
            for i in range(Np):
                # Off-diagonal 2D marginalized plots
                if (j>i):
                    # 1-sigma ellipse
                    errorellipse, xyaxes=self.error_ellipse(covariance_matrix, i,j, nstd=nstd, clr=clr, alpha=0.75, zorder=0)
                    # 2-sigma ellipse
                    ere2, xyaxes2 = self.error_ellipse(covariance_matrix, i, j, nstd=2*nstd, clr=clr, alpha=0.35, zorder=1)
                    
                    # Define axes
                    jp=i
                    ip=j
                    axis=allaxes[ip][jp]
                    axis.locator_params(axis='x', nbins=nbinsx, min_n_ticks=3)
                    axis.locator_params(axis='y', nbins=nbinsy, min_n_ticks=3)
                    axis.add_artist(ere2)
                    axis.add_artist(errorellipse)
                    axis.axis(xyaxes2)
                    if (i==0):
                        axis.set_ylabel(self.paramLaTeXNames()[j], fontsize=14)
                    if (i>=0):
                        axis.set_xlabel(self.paramLaTeXNames()[i], fontsize=14)

                    # Make ticks visible and make sure they are not too crowded
                    axis.ticklabel_format(useOffset=False)
                    axis.tick_params(axis='both', which='both', direction='in', 
                                     top=True, right=True)
                    for label in axis.get_xticklabels():
                        label.set_rotation(90) 
                    axis.tick_params(labelsize=8)                       # Tick values
                    axis.yaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    axis.xaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    if i != 0:
                        for tick in axis.yaxis.get_major_ticks():
                            tick.label1.set_visible(False)

                    axis.plot(self.fiducialParams()[i],self.fiducialParams()[j],zorder=2,
                              marker='P',markerfacecolor='white',markeredgecolor='k')

                    
                # On-diagonal 1D marginalized plots
                if (j==i):
                    y, x, xyaxes=self.oneD_constraints(covariance_matrix, i,j, clr=clr)
                    
                    jp=i
                    axis=allaxes[jp][jp]

                    axis.locator_params(axis='x', nbins=nbinsx, min_n_ticks=3)
                    axis.locator_params(axis='y', nbins=nbinsy, min_n_ticks=3)
                    
                    axis.axis(xyaxes)
                    
                    if (i==0):
                        axis.set_ylabel(self.paramLaTeXNames()[j], fontsize=14)
                    if (i>=0):
                        axis.set_xlabel(self.paramLaTeXNames()[i], fontsize=14)

                    axis.tick_params(axis='both', which='both', direction='in',
                                     top=True, right=True)

                    for label in axis.get_xticklabels():
                        label.set_rotation(90) 
                    axis.tick_params(labelsize=8)                       # Tick values
                    axis.yaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    axis.xaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    if i != 0:
                        for tick in axis.yaxis.get_major_ticks():
                            tick.label1.set_visible(False)
                            
                    if i == Np-1:
                        mu=np.mean(self.pmat[Np-1])
                        sig=self.error(covariance_matrix)[Np-1]
                        axis.set_xlim(mu-2.8*sig,mu+2.8*sig)

                    axis.plot(x,y,color=clr)
        
        # Hide empty plots above the diagonal
        for i in range(Np):
            for j in range(Np):
                if (j>i):
                    allaxes[i][j].axis('off')
                
        # Tight layout
        plt.tight_layout()

        # Remove whitespace between subplots
        plt.subplots_adjust(wspace=0, hspace=0)

        f.align_xlabels(allaxes[-1])
        f.align_ylabels(allaxes[:,0])

        # Save figure
        if figname == None:
            figname = self.path_to_col+'covariance_matrix_triangle_plot'
        plt.savefig(figname+'.pdf', format='pdf')
        #plt.savefig(figname+'.PNG', format='png', dpi=1000)  
        #plt.show()

    def plot_error_matrix_combined(self, covariance_matrix_array, filename=None, labels=None, nstd=1, nbinsx=6, nbinsy=6, color=False):
        """Create a triangle plot for comparison of multiple experiments
        Some examples:
        This allows one to plot Experiment A in one color versus Exp. B in another color
        This also allows one to plot (Exp. A + Exp. B) versus Exp. A
        """
        self.params()
        Np=self.N_pars
        
        if color==True:
            clrs = ["cornflowerblue", "g", "r", "g", "r", "r"]
        else:
            clrs = ['k','gray']
        linestyles = ['solid', 'solid', 'solid', 'solid']
        linewidths = [1.5, 2.0, 2.5, 1.5]
        fills = [True, True, True, True]
        alphas = [1.0, 0.3, 1.0, 1.0]

        f, allaxes = plt.subplots(Np, Np, sharex='col', figsize=(self.N_pars,self.N_pars))

        for j in range(Np):
            for i in range(Np):
                if (j>i):
                    cmcnt=0
                    for c in range(len((covariance_matrix_array))):
                        errorellipse, xyaxes=self.error_ellipse(covariance_matrix_array[c], i,j, 
                                                                nstd=nstd, clr=clrs[cmcnt], alpha=alphas[cmcnt])
                        ere2, xyaxes2 = self.error_ellipse(covariance_matrix_array[c], i, j, 
                                                           nstd=2*nstd, clr=clrs[cmcnt], alpha=alphas[cmcnt]/2.)
                        errorellipse.set_linestyle(linestyles[cmcnt])
                        errorellipse.set_linewidth(linewidths[cmcnt])
                        errorellipse.set_fill(fills[cmcnt])
                        ere2.set_linestyle(linestyles[cmcnt])
                        ere2.set_linewidth(linewidths[cmcnt])
                        ere2.set_fill(fills[cmcnt])
                        
                        jp=i
                        ip=j
                        axis=allaxes[ip][jp]
                        
                        axis.add_artist(ere2)
                        axis.add_artist(errorellipse)
                        axis.axis(xyaxes2)
                        if cmcnt == 0:
                            axis.axis(xyaxes)
                        cmcnt=cmcnt+1
                        
                    axis.locator_params(axis='x', nbins=nbinsx)
                    axis.locator_params(axis='y', nbins=nbinsy)
                    if (i==0):
                        axis.set_ylabel(self.paramLaTeXNames()[j], fontsize=14)
                    if (i>=0):
                        axis.set_xlabel(self.paramLaTeXNames()[i], fontsize=14)
                    
                    axis.ticklabel_format(useOffset=False)
                    axis.tick_params(axis='both', which='both', direction='in', 
                                     top=True, right=True)
                    for label in axis.get_xticklabels():
                        label.set_rotation(-90) 
                    axis.tick_params(labelsize=8)                       # Tick values
                    axis.yaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    axis.xaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    if i != 0:
                        for tick in axis.yaxis.get_major_ticks():
                            tick.label1.set_visible(False)

                if (j==i):
                    cmcnt=0
                    for c in range(len((covariance_matrix_array))):
                        y, x, xyaxes=self.oneD_constraints(covariance_matrix_array[c], i,j)
                        jp=i
                        axis=allaxes[jp][jp]
                        axis.plot(x,y,color=clrs[cmcnt],alpha=alphas[cmcnt],label=r'$n=\,\,$'+labels[cmcnt])

                        cmcnt=cmcnt+1

                    axis.locator_params(axis='x', nbins=nbinsx)
                    axis.locator_params(axis='y', nbins=nbinsy)
        
                    axis.axis(xyaxes)

                    if (i==0):
                        axis.set_ylabel(self.paramLaTeXNames()[j], fontsize=14)
                    if (i>=0):
                        axis.set_xlabel(self.paramLaTeXNames()[i], fontsize=14)

                    axis.tick_params(axis='both', which='both', direction='in',
                                     top=True, right=True)

                    for label in axis.get_xticklabels():
                        label.set_rotation(-90) 
                    axis.tick_params(labelsize=8)                       # Tick values
                    axis.yaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    axis.xaxis.get_offset_text().set_fontsize(8)        # Scientific notation floats
                    if i != 0:
                        for tick in axis.yaxis.get_major_ticks():
                            tick.label1.set_visible(False)
                            
                    if i == Np-1:
                        axis.relim()
                        axis.autoscale_view()


        for i in range(Np):
            for j in range(Np):
                if (j>i):
                    allaxes[i][j].axis('off')
                    legend=axis.legend(framealpha=0,bbox_to_anchor=(-0.5, 7.05),fontsize=14)


        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        f.align_xlabels(allaxes[-1])
        f.align_ylabels(allaxes[:,0])

        #Save figure
        plt.savefig(filename+'.pdf', format='pdf') 
        #plt.show()        
        

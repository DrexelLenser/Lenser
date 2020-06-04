"""
Module: lenser_galaxy
.. synopsis: Holds a real galaxy image, or a model galaxy image and model parameters
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>
"""



import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
from scipy import optimize
import pickle
import pandas as pd
import glob as glob
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

class Galaxy(object):

    """
    Galaxy class: 
    .. Holds the list of parameters used in the modified S\'ersic model.
    .. There are six galaxy shape parameters and (up to) seven lens parameters:
    .. .. p = {xc,yc,ns,rs,q,phi,psi11,psi12,psi22,psi111,psi112,psi122,psi222}
    .. Galaxy().generateImage() function:
    .. .. Holds the modified S\'ersic model
    .. .. Points to the Lens class and performs the lensing coordinate deprojection
    .. .. Points to the Image class to create a two-dimensional image of the model
    .. .. Performs PSF convolution if a PSF is available
    """

    def __init__(self, xc=0., yc=0., ns=0.5, rs=1., q=1., phi=0., galaxyLens=None):
        self.name=''
        self.xc = xc
        self.yc = yc
        self.ns = ns
        self.rs = rs
        self.q = q
        self.phi = phi
        self.galaxyLens = galaxyLens

    def setPar(self, val, type):
        """ 
        Set the value for each galaxy shape parameter
        .. xc: centroid x coordinate
        .. yc: centroid y coordinate
        .. ns: seric index
        .. rs: characteristic radius
        .. q: semi-major-to-semi-minor axis ratio
        .. phi: orientation angle
        """
        if (type == 'xc'):
            self.xc = val
        elif (type == 'yc'):
            self.yc = val
        elif (type == 'ns'):
            self.ns = val
        elif (type == 'rs'):
            self.rs = val
        elif (type == 'q'):
            self.q = val
        elif (type == 'phi'):
            self.phi = val

    def setPsi2(self, psi2new):
        """
        Set the values for the psi2 lensing array
        .. These include the covergence and shear terms
        .. psi2 = [psi11, psi12, psi22]
        """
        self.galaxyLens.setPsi2(psi2new)

    def setPsi3(self, psi3new):
        """
        Set the values for the psi3 lensing array
        .. These include the flexion terms
        .. psi3 = [psi111, psi112, psi122, psi222]
        """
        self.galaxyLens.setPsi3(psi3new)

    def setLens(self, newlens):
        """
        Set the entire lensing array with one call
        ... psi = [psi2, psi3]
        """
        self.galaxyLens = newlens

    def generateImage(self, nx, ny, lens=False, I0=1., 
                        noise1=0, noise2=0, seed=None, 
                        background=0, psfmap=None):
        """
        Create a galaxy image
        .. Holds the modified S\'ersic model
        .. Points to the Lens class and performs the lensing coordinate deprojection
        .. Points to the Image class to create a two-dimensional image of the model
        .. Performs PSF convolution if a PSF is available
        """
                
        # Lay down the coordinate system 
        # .. The coordinates correspond to the bottom-left of each pixel
        # .. The coordinate system's origin is the centroid of the galaxy
        y0,x0 = np.mgrid[0:nx,0:ny] 
        delx = nx/2
        dely = ny/2

        # theta: the lens-plane coordinates
        thetax = x0-self.xc-0.5-delx 
        thetay = y0-self.yc-0.5-dely

        # Deproject theta --> beta: the source-plane coordinates
        if (lens == True) and (self.galaxyLens != None):
            betax,betay = self.galaxyLens.deproject(thetax,thetay)
        else:
            betax = thetax
            betay = thetay

        # x and y are the centroid-subtracted source-plane coordinates rotated 
        # appropriately by an orientation angle phi
        x = betax*np.cos(self.phi)+betay*np.sin(self.phi)
        y = betay*np.cos(self.phi)-betax*np.sin(self.phi)

        # theta_prime is the radial coordinate, 
        # and q is the semimajor-to-semiminoraxis ratio of the galaxy
        theta_prime = np.sqrt(x**2/self.q**2+y**2)

        # Create the galaxy model datamap using the modified S\'ersic intensity profile 
        # .. Note that the model is normalized to have I0=1 initially
        datamap = np.exp(-(theta_prime/self.rs)**(1/self.ns))

        # If a PSF is present, convolve it with the galaxy model
        if psfmap is not None:
            datamap = convolve_fft(datamap,psfmap)

        # Multiply the datamap by I0
        datamap = I0*datamap
        
        # If you wish to simulate a realistic galaxy image, 
        # (as opposed to using this function to simply create a galaxy model),
        # you likely want to generate some random noise and add a background
        # the noise is generated via noise=random.normal(0,sqrt(noise1**2+noise2**2*f))
        # .. note that by default, noise = 0.  
        # .. note that by default, background = 0
        # .. You should use these default valuesiIf you are creating a galaxy model
        # .. as opposed to simulating a galaxy.
        if seed != None:
            np.random.seed(seed)
        noisemap = np.sqrt(noise1**2+noise2**2*abs(datamap))
        datamap = datamap+noisemap*np.random.normal(size=datamap.shape)+background

        # Create an Image of this galaxy
        myImage = Image(self.name,datamap,noisemap)
        return myImage



def fits_read(name):

    """
    fits_read function:
    .. Reads in FITS file for galaxy postage stamp.   
    .. Additionally, looks for FITS files for a noisemap, segmentation map, 
       and PSFmap for the corresponding galaxy.
    .. If a noisemap is not provided, fits_read will search for a pickle file at 
       the location '../*noise-info.pkl' that contains information in order to 
       calculate one.
    .. If this pickle file does not exists, then one is calculated based on simple 
       assumptions in Image()
    """

    #Read in galaxy postage stamp
    data_file = fits.open(name)[0].data

    # Look for noisemap, segmentation map, and PSFmap.  Check to see if the galaxy 
    # postage stamp name contains a specific visual band.
    if '_u' in name.split('.fits')[0]:
        band='u'
    elif '_g' in name.split('.fits')[0]:
        band='g'
    elif '_r' in name.split('.fits')[0]:
        band='r'
    elif '_i' in name.split('.fits')[0]:
        band='i'
    elif '_z' in name.split('.fits')[0]:
        band='z'
    else:
        band=None
    if band != None:
        rms = name.split('_r.fits')[0]+'_'+band+'_rms.fits'
        seg = name.split('_r.fits')[0]+'_'+band+'_seg.fits'
        psf = name.split('_r.fits')[0]+'_'+band+'_psf.fits'
    else:
        rms = name.split('.fits')[0]+'_rms.fits'
        seg = name.split('.fits')[0]+'_seg.fits'
        psf = name.split('.fits')[0]+'_psf.fits'

    # Create a segmentation mask from the segmentation map
    # .. The segmentation map is an array of integers, where a single integer is assigned for every
    # .. pixel of the postage stamp. These maps are typically generated using Source Extractor, which seeks
    # .. to identify the background (and assign it a value 0); the main galaxy object (value 1); and
    # .. any other foreground objects that may be present in the postage stamp (value 2 for the first
    # .. extraneous foreground object, 3 for the second, etc.), such as stars, bad pixels, etc. Lenser
    # .. then creates a segmentation mask, wherein the background and galaxy get values of 1, and
    # .. all other foreground objects get a value of 0 – i.e., they are masked out. 
    if os.path.exists(seg):

        # Read in segmentation map
        seg_map = fits.open(seg)[0].data
        seg_mask = seg_map*0
        id0 = np.where((seg_map == 0))
        id1 = np.where((seg_map == 1))

        # Define segmentaton mask
        seg_mask[id0] = 1
        seg_mask[id1] = 1

        # Background mask corresponds to the background pixels only
        bg_mask = seg_map*0
        bg_mask[id0] = 1

        # Object mask corresponds to the object pixels only
        obj_mask = seg_map*0
        obj_mask[id1] = 1
        
        # Run friends-of-friends algorithm to ensure that all pixels in the object mask are contiguous.
        # First get rid of pixels near the edges with obj_mask=1
        for k in range(6):
            for i in range(obj_mask.shape[0]):
                for j in range(obj_mask.shape[0]):
                    if obj_mask[i][j] == 1:
                        if i+(k+1) > obj_mask.shape[0]:
                             obj_mask[i][j] = 0
                        elif i-(k+1) < 0:
                             obj_mask[i][j] = 0
                        elif j+(k+1) > obj_mask.shape[1]:
                             obj_mask[i][j] = 0
                        elif j-(k+1) < 0:
                             obj_mask[i][j] = 0
        # Now require contiguity
        for k in range(5):
            for i in range(obj_mask.shape[0]):
                for j in range(obj_mask.shape[0]):
                    if obj_mask[i][j] == 1:
                        if obj_mask[i-(k+1)][j] == 1:
                             obj_mask[i][j] = 1
                        elif obj_mask[i+(k+1)][j] == 1:
                             obj_mask[i][j] = 1
                        elif obj_mask[i][j-(k+1)] == 1:
                             obj_mask[i][j] = 1
                        elif obj_mask[i][j+(k+1)] == 1:
                             obj_mask[i][j] = 1
                        else:
                            obj_mask[i][j] = 0

        # Define final segmentation mask after contiguity check
        id0 = np.where((seg_map == 0))
        id1 = np.where((obj_mask == 1))
        seg_mask = seg_map*0
        seg_mask[id0] = 1
        seg_mask[id1] = 1
    else:
        # If a segmentation map is absent, the assumption is that the input postage stamp includes only 
        # the background and galaxy, and hence all pixels are viable.
        seg_mask = np.ones(data_file.shape)
        obj_mask = None
        bg_mask = None

    # Noisemap
    if os.path.exists(rms):
        rms_file = fits.open(rms)[0].data
    else:
        rms_file = None
 
    # Now search for a .pkl file containing noise info.
    # The .pkl filename must be located in the same directory of
    # the input image, e.g. '*noise-info_r.pkl'
    # where the 'r' extension is an e.g. where this information is
    # specifically for the r band.  The .pkl file must contain 
    # a pandas dataframe and have (at least) the following columns
    # (we will again consider the e.g. of the 'r' band:)
    #  name    gain_r    sky_r    skyErr_r    darkVariance_r
    if rms_file is None:
        gal_name_w_band = name.split('/')[-1].split('.fits')[0]
        path = name.split(gal_name_w_band)[0]
        gal_name = gal_name_w_band.split('_r')[0]

        # Get noise info file
        noise_info_file = glob.glob(path+'*noise-info_'+band+'.pkl')[0]

        # Get information from noise info file
        if os.path.exists(noise_info_file):
            noise_info = pickle.load(open(noise_info_file, 'rb'),encoding='latin1')
            idx = None
            for i in range(len(noise_info)):
                if noise_info['name'][i] == gal_name:
                    idx = i
            gain = noise_info['gain_'+band][idx]
            sky = noise_info['sky_'+band][idx]
    else:
        gain = None
        sky = None
        
    # PSFmap
    if os.path.exists(psf):
        psf_file = fits.open(psf)[0].data
    else:
        psf_file = None

    return data_file, rms_file, seg_mask, bg_mask, psf_file, gain, sky



class Image(object):

    """
    Image class:
    .. Holds various two-dimensional arrays referred to as "maps"
    .. .. datamap: 
    .. .. .. Corresponds to the galaxy image.  
    .. .. .. Can either by a real galaxy image from a FITS file, read in from fits_read(), 
             or it can be a model galaxy image, generated by Galaxy().generateImage()
    .. .. noisemap: 
    .. .. .. Noise in the galaxy image.  
    .. .. .. Can either be a real noisemap from a FITS file, read in from fits_read(),
             or, in the absense of a noisemap, Image() generates one.
    .. .. psfmap: 
    .. .. .. PSF read in from fits_read()
    .. .. .. If one is not provided, PSF convolution is ignored throughout Lenser.
    .. Holds various two-dimensional arrays referred to as "masks"
    .. .. segmentation mask:
    .. .. .. see desciption in fits_read()
    .. .. elliptical mask:
    .. .. .. Generated so as to include only relevant pixels in the input image, reducing error from sources
             near the edge of the postage stamp. During this process, we also estimate: (i). the background map 
             and (ii). the noisemap, in the case that a noisemap is not already provided and read in through 
             fits_read(). The background is then subtraced from the datamap. 
    """

    def __init__(self, name, datamap=None, noisemap=None, maskmap=None, segmask=None, psfmap=None):
        """
        If a datamap is provided, you are dealing with a galaxy model.
        If a datamap is not provided, then you are dealing with a real galaxy image, 
        and it is imported from fits_read()
        """
        if datamap is not None:
            self.name = name
            self.datamap = datamap
            self.noisemap = noisemap
            self.maskmap = maskmap
            self.segmask = segmask
            self.psfmap = psfmap
        else:
            data, noise, seg, bg, psf, gain, sky = fits_read(name)
            name = name.split('/')[-1].split('.fits')[0]
            self.name = name
            self.datamap = data
            self.noisemap = noise
            self.maskmap = maskmap
            self.segmask = seg
            self.bgmask = bg
            self.psfmap = psf
            #self.psfmap = None
            self.gain = gain
            self.sky = sky
        # Dimensions of datamap
        self.nx = self.datamap.shape[0]
        self.ny = self.datamap.shape[1]

    def getName(self):
        """
        Get string of the galaxy name
        """
        return self.name

    def getLaTeXName(self):
        """
        Get galaxy name in LaTeX formatting
        """
        return r'${\rm '+self.name.replace('_',r'~')+'}$'

    def plot(self, type='data', show=False, save=True):
        """
         Plot individual maps. 
         .. We multiply the datamap by the segmask for plotting purposes only, 
            for better visualization (otherwise extraneous pixels have overpowering brightness).
        """
        if (type == 'data'):
            plt.imshow(np.flipud(self.datamap)*np.flipud(self.segmask),cmap='gray',origin='lower')
            plt.title(self.getLaTeXName())
        elif (type == 'mask'):
            plt.imshow(np.flipud(self.maskmap),cmap='gray',origin='lower')
            plt.title(self.getLaTeXName()+' '+r'${\rm mask~map}$')
        elif (type == 'noise'):
            if self.noisemap is not None:
                plt.imshow(np.flipud(self.noisemap),cmap='gray',origin='lower')
                plt.title(self.getLaTeXName()+' '+r'${\rm noise~map}$')
        elif (type == 'psf'):
            if self.psfmap is not None:
                plt.imshow(np.flipud(self.psfmap),cmap='gray',origin='lower')
                plt.title(self.getLaTeXName()+' '+r'${\rm PSF~map}$')
        elif (type == 'noise_masked'):
            plt.imshow(np.flipud(self.noisemap)*np.flipud(self.maskmap),cmap='gray',origin='lower')
            plt.title(self.getLaTeXName()+' '+r'${\rm masked~noise~map}$')
        if save == True:
            plt.savefig(self.getName()+'_'+type+'.pdf', format='pdf')
        if show == True:
            plt.show()

    def getMap(self, type='data'):
        """
        Return various maps
        """
        if type == 'data':
            return self.datamap
        elif type == 'mask':
            return self.maskmap
        elif type == 'noise':
            return self.noisemap
        elif type == 'segmask':
            return self.segmask
        elif type == 'bgmask':
            return self.bgmask
        elif type == 'psf':
            return self.psfmap
        elif type == 'noise_masked':
            return self.noisemap*self.maskmap

    def setMap(self, newdata, type='data'):
        """ 
        Set a new map
        .. This function will not check for correct map shape
        """
        if type == 'data':
            self.datamap=newdata
        elif type == 'mask':
            self.maskmap=newdata
        elif type == 'noise':
            self.noisemap=newdata
        elif type == 'segmask':
            self.segmask=newdata
        elif type == 'bgmask':
            self.bgmask=newdata
        elif type == 'psf':
            self.psfmap=newdata

    def getMoments(self, Qijkl, id=None):
        """
        Get the (n + m) image moments <x^n y^m>.
        The moments, up to order four, are
        .. Order 1: <x>, <y>
        .. Order 2: <xx>, <xy>, <yy>
        .. Order 3: <xxx>, <xxy>, <xyy>, <yyy>
        .. Order 4: <xxxx>, <xxxy>, <xxyy>, <xyyy>, <yyyy>
        """
        # Lay down the coordinate system 
        # .. The coordinates correspond to the bottom-left of each pixel
        y,x = np.mgrid[0:self.nx,0:self.ny] 
        x = x-0.5
        y = y-0.5

        centroid = np.zeros(2)
        order1 = np.zeros(2)
        order2 = np.zeros(3) 
        order3 = np.zeros(4)
        order4 = np.zeros(5)

        if id == None:
            f0 = np.sum(self.datamap*self.maskmap*self.segmask)
            centroid[0] = np.sum(self.datamap*x*self.maskmap*self.segmask)/f0 
            centroid[1] = np.sum(self.datamap*y*self.maskmap*self.segmask)/f0 
            dx = x-centroid[0]
            dy = y-centroid[1]
            for idx in range(2):
                order1[idx] = np.sum(self.datamap*pow(dx,1-idx)*pow(dy,idx)*self.maskmap*self.segmask)/f0
            for idx in range(3):
                order2[idx] = np.sum(self.datamap*pow(dx,2-idx)*pow(dy,idx)*self.maskmap*self.segmask)/f0
            for idx in range(4):
                order3[idx] = np.sum(self.datamap*pow(dx,3-idx)*pow(dy,idx)*self.maskmap*self.segmask)/f0
            for idx in range(5):
                order4[idx] = np.sum(self.datamap*pow(dx,4-idx)*pow(dy,idx)*self.maskmap*self.segmask)/f0
        elif id != None:
            f0 = np.sum(self.datamap[id])
            centroid[0] = np.sum(self.datamap[id]*x[id])/f0 
            centroid[1] = np.sum(self.datamap[id]*y[id])/f0 
            dx = x[id]-centroid[0]
            dy = y[id]-centroid[1]
            for idx in range(2):
                order1[idx] = np.sum(self.datamap[id]*pow(dx,1-idx)*pow(dy,idx))/f0
            for idx in range(3):
                order2[idx] = np.sum(self.datamap[id]*pow(dx,2-idx)*pow(dy,idx))/f0
            for idx in range(4):
                order3[idx] = np.sum(self.datamap[id]*pow(dx,3-idx)*pow(dy,idx))/f0
            for idx in range(5):
                order4[idx] = np.sum(self.datamap[id]*pow(dx,4-idx)*pow(dy,idx))/f0
        
        if Qijkl == 'f0':
            return f0
        if Qijkl == 'x':
            return x
        if Qijkl == 'y':
            return y
        if Qijkl == 'xc':
            return centroid[0]
        if Qijkl == 'yc':
            return centroid[1]
        if Qijkl == 'Q1':
            return order1[0]
        if Qijkl == 'Q2':
            return order1[1]
        if Qijkl == 'Q11':
            return order2[0]
        if Qijkl == 'Q12':
            return order2[1]
        if Qijkl == 'Q22':
            return order2[2]
        if Qijkl == 'Q111':
            return order3[0]
        if Qijkl == 'Q112':
            return order3[1]
        if Qijkl == 'Q122':
            return order3[2]
        if Qijkl == 'Q222':
            return order3[3]
        if Qijkl == 'Q1111':
            return order4[0]
        if Qijkl == 'Q1112':
            return order4[1]
        if Qijkl == 'Q1122':
            return order4[2]
        if Qijkl == 'Q1222':
            return order4[3]
        if Qijkl == 'Q2222':
            return order4[4]
        if Qijkl == 'all':
            return f0, x, y, centroid, order1, order2, order3, order4
        if Qijkl == 'x,y,centroid,order2':
            return x, y, centroid, order2
        if Qijkl == 'centroid,order2':
            return centroid, order2
        if Qijkl == 'order2':
            return order2
        if Qijkl =='order2,order3,order4':
            return order2, order3, order4


    def generateMask(self, subtractBackground=True):
        """
        Here we generate the elliptical mask.
        During this process, we also estimate:
          (i). the background map
          (ii). the noisemap, in the case that a noisemap is not already provided
                to Lenser and read in through fits_read().
        The background is then subtraced from the datamap.  The background is not
        itself a global variable.
        """
        # Background calculation if a segmentation map is provided
        if self.bgmask is not None:

            # Get background pixels and their corresponding indices
            id_bg = np.where(self.bgmask==1)
            bg_pix = self.datamap[id_bg]

            # Lay down the coordinate system 
            # .. The coordinates correspond to the bottom-left of each pixel
            y,x = np.mgrid[0:self.nx,0:self.ny] 
            x = x-0.5
            y = y-0.5

            # Functional form of a background with z-intercept bg0, and slope in x and y
            def bgGradient(coords, bg0, mx, my):
                x = coords[0]
                y = coords[1]
                return (bg0 + mx*(x-self.nx/2.) + my*(y-self.ny/2.))

            # Do a best fit for the background map
            bg0_guess = np.median(self.datamap[id_bg])
            mx_guess = 0.
            my_guess = 0.
            pars_guess = np.asarray([bg0_guess, mx_guess, my_guess])

            x_id_bg = x[id_bg]
            y_id_bg = y[id_bg]
            coords_id_bg = np.array((x_id_bg, y_id_bg))
            popt,pcov = optimize.curve_fit(bgGradient, coords_id_bg, bg_pix, pars_guess)

            coords = np.array((x,y))
            bg = bgGradient(coords,*popt)
        
        # Background calculation if a segmentation map is not provided to Lenser
        elif self.bgmask is None:

            # Since we do not have a background mask, we want to create one of sorts.
            # We will create a mask that goes around the entire edge of the postage
            # stamp and has a thickness of 10% of the width and height of the stamp
            # for the thicknesses in the x and y direction, respecitvely.
            bg_mask = np.zeros(self.datamap.shape)
            # 10% buffer
            xbuf = int(0.1*self.nx)
            ybuf = int(0.1*self.ny)
            # Bins at top of stamp:
            bg_mask[0:ybuf] = 1
            # Bins at bottom of the stamp
            bg_mask[-(1+ybuf):] = 1
            # Bins at the left of the stamp
            bg_mask[:,0:xbuf] = 1
            # Bins at the right of the stamp
            bg_mask[:,-(1+xbuf):] = 1
            # Background mask
            self.bgmask=bg_mask

            # Get background pixels and their corresponding indices
            id_bg = np.where(self.bgmask ==1)
            bg_pix = self.datamap[id_bg]

            # Lay down the coordinate system 
            # .. The coordinates correspond to the bottom-left of each pixel
            y,x = np.mgrid[0:self.nx,0:self.ny] 
            x = x-0.5
            y = y-0.5

            # Functional form of a background with z-intercept bg0, and slope in x and y
            def bgGradient(coords, bg0, mx, my):
                x = coords[0]
                y = coords[1]
                return (bg0 + mx*(x-self.nx/2.) + my*(y-self.ny/2.))

            # Do a best fit for the background map
            bg0_guess = np.median(self.datamap[id_bg])
            mx_guess = 0.
            my_guess = 0.
            pars_guess = np.asarray([bg0_guess, mx_guess, my_guess])

            x_id_bg = x[id_bg]
            y_id_bg = y[id_bg]
            coords_id_bg = np.array((x_id_bg,y_id_bg))
            popt,pcov = optimize.curve_fit(bgGradient, coords_id_bg, bg_pix, pars_guess)

            coords = np.array((x,y))
            bg = bgGradient(coords,*popt)

        # Subtract the background from the datamap (if requested)
        if (subtractBackground == True):
            self.datamap = self.datamap-bg
        
        
        # Calculate noisemap (if one is not provided)
        # .. The assumption here is that the noise contributions are a flat sky noise and
        # .. a Poisson noise
        if self.noisemap is None:
            if self.gain is None:
                id_bg = np.where(self.bgmask==1)
                noise1 = np.ma.std(self.datamap[id_bg])*np.ones(self.datamap.shape)
                noise2 = np.sqrt(abs(self.datamap*self.segmask))
                self.noisemap = np.sqrt(noise1**2.+noise2**2.)
            else:
                counts = self.datamap*self.segmask
                sky = self.sky*self.segmask
                id_bg = np.where(self.bgmask==1)
                noise1 = np.ma.std(self.datamap[id_bg])*np.ones(self.datamap.shape)
                noise2 = np.sqrt(abs(counts+sky+bg)/self.gain)
                self.noisemap = np.sqrt(noise1**2.+noise2**2.)
                
        # Calculate the elliptical mask
        # .. nsig is a heuristical number
        nsig = 2.5

        id=np.where(abs(self.datamap*self.segmask) > abs(nsig*self.noisemap*self.segmask))
        for i in range(3):
            x, y, centroid, order2 = self.getMoments('x,y,centroid,order2', id)
            xc = centroid[0]
            yc = centroid[1]
            Q11 = order2[0]
            Q12 = order2[1]
            Q22 = order2[2]

            chi1 = (Q11-Q22)/(Q11+Q22)
            chi2 = 2*Q12/(Q11+Q22)
            chisq = chi1**2+chi2**2
            phi = np.arctan2(chi2,chi1)/2.

            q = np.sqrt((1+np.sqrt(chisq))/(1-np.sqrt(chisq)))
            x1 = (x-xc)*np.cos(phi)+(y-yc)*np.sin(phi)
            y1 = (y-yc)*np.cos(phi)-(x-xc)*np.sin(phi)

            # Elliptical mask:
            id=np.where((x1/np.sqrt(1+chisq))**2+(y1/np.sqrt(1-chisq))**2 < nsig**2*(Q11+Q22))

        self.maskmap = np.zeros(self.datamap.shape)
        self.maskmap[id] = 1



class Lens(object):

    """
    Lens class:
    .. Handles the lensing coordinate deprojection
    .. Temporarily holds the (up to) seven lens parameters before they are passed into the Galaxy class.
    """

    def __init__(self, psi2=[0,0,0], psi3=[0,0,0,0]):
        self.psi2 = psi2
        self.psi3 = psi3

    def deproject(self, thetax, thetay):
        """
        Lensing coordinate deprojection
        .. Note: thetax, thetay can be numbers or numpy objects.
        """
        betax = thetax
        betay = thetay
        fact = [0.5,1,0.5]
        for i in range(2):
            betax = betax-self.psi2[i]*thetax**(1-i)*thetay**(i)
            betay = betay-self.psi2[i+1]*thetax**(1-i)*thetay**(i)
        for i in range(3):
            betax = betax-fact[i]*self.psi3[i]*thetax**(2-i)*thetay**(i)
            betay = betay-fact[i]*self.psi3[i+1]*thetax**(2-i)*thetay**(i)
        return betax, betay

    def setPsi2(self, psi2new):
        self.psi2 = psi2new

    def setPsi3(self, psi3new):
        self.psi3 = psi3new

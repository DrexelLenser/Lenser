"""
Module: lenser_galaxy
.. synopsis: Holds a real galaxy image, or a model galaxy image and model parameters
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>
"""



import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
from scipy import optimize
import pandas as pd
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
    .. .. Points to the Lens() class and performs the lensing coordinate deprojection
    .. .. Points to the Image() class to create a two-dimensional image of the model
    .. .. Performs PSF convolution if a PSF is available
    """

    def __init__(self, xc=0., yc=0., ns=0.5, rs=1., q=1., phi=0., galaxyLens=None, galaxyQuadrupole=None):
        self.name=''
        self.xc = xc
        self.yc = yc
        self.ns = ns
        self.rs = rs
        self.q = q
        self.phi = phi
        self.galaxyLens = galaxyLens
        self.galaxyQuadrupole = galaxyQuadrupole

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

    def setQuadrupole(self, quadrupolenew):
        """
        Set the quadrupole moments Q_ij for an Image (or MultiImage, averaged over all
        available epochs and bands) for ease of access, so they only need to be calculated
        once rather than calling Image().getMoments() multiple times
          quadrupolenew = (Q11, Q12, Q22)
        """
        self.GalaxyQuadrupole = quadrupolenew

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
            datamap = convolve_fft(datamap, psfmap,
                                   normalize_kernel = False, psf_pad = False,
                                   nan_treatment = 'fill')#, fft_pad = False )

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

    

class Lens(object):

    """
    Lens class:
    .. Handles the lensing coordinate deprojection
    .. Temporarily holds the (up to) seven lens parameters before they are passed into the Galaxy() class.
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

        

class Image(object):

    """
    Image class:
    .. Holds various two-dimensional arrays referred to as "maps"
    .. .. datamap: 
    .. .. .. Corresponds to the science image postage stamp of a galaxy.  
    .. .. .. Can either by a real galaxy image from a FITS file, which can be handled with lenser_fits.py, 
             or it can be a model galaxy image, generated by Galaxy().generateImage()
    .. .. noisemap: 
    .. .. .. rms noise in the galaxy image.  
    .. .. .. Can either be a real noisemap from a FITS file, which can be handled with lenser_fits.py,
             or, in the absense of a noisemap, Image() generates one.
    .. .. weightmap:
    .. .. .. Inverse variance (1/noisemap**2) weighting of image noise
    .. .. .. NOTE: One should only supply either a noisemap or a weightmap
    .. .. psfmap: 
    .. .. .. Point-spread function (PSF) associated with galaxy image.
    .. .. .. If one is not provided, PSF convolution is ignored throughout Lenser.
    .. Holds various two-dimensional arrays referred to as "masks"
    .. .. segmentation mask:
    .. .. .. Obtained from the SExtractor segmentation map and lenser_fits.py
    .. .. .. Bitmask where all non-galaxy pixels = 0 (galaxy pixels and background pixels = 1)
    .. .. background mask:
    .. .. .. Obtained from the SExtractor segmentation map and lenser_fits.py
    .. .. .. Bitmask where only background pixels = 1
    .. .. object mask (optional):
    .. .. .. Obtained from the SExtractor segmentation map and lenser_fits.py
    .. .. .. Bitmask where only galaxy pixels = 1   
    .. .. ubersegmentation mask (optional):
    .. .. .. Obtained from lenser_fits.py
    .. .. .. Bitmask where any pixel that is closer to another object than the galaxy = 0
    .. .. weighted ubersegmentation mask (optional):
    .. .. .. weightmap multiplied by the ubersegmentation mask
    .. .. elliptical mask:
    .. .. .. Generated so as to include only relevant pixels in the input image, reducing error from sources
             near the edge of the postage stamp. During this process, we also estimate: (i). the background map 
             and (ii). the noisemap, in the case that a noisemap is not already provided. 
             Option to subtract the background from the datamap. 
    """

    def __init__(self, name=None, datamap=None,
                 noisemap=None, wtmap=None,
                 ellipmask=None, segmask=None,
                 ubersegmask=None, wtubersegmask=None,
                 bgmask=None, objmask=None,
                 psfmap=None,
                 gain=None, sky=None):

        self.name = name
        self.datamap = datamap
        self.noisemap = noisemap
        self.wtmap = wtmap
        self.ellipmask = ellipmask
        self.segmask = segmask
        self.ubersegmask = ubersegmask
        self.wtubersegmask = wtubersegmask
        self.bgmask = bgmask
        self.objmask = objmask
        self.psfmap = psfmap

        # Optional, in case where noisemaps or weightmaps are not provided:
        self.gain = gain                       # Gain in galaxy image
        self.sky = sky                         # Sky-level in galaxy image

        # Weighted mask, calculated for convinence for use in lenser_aim.py
        self.wtMask = None
        
        # Dimensions of datamap
        if datamap is not None:
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

    def getMap(self, type='data'):
        """
        Return various maps
        """
        if type == 'data':
            return self.datamap
        elif type == 'noise':
            return self.noisemap
        elif type == 'wt':
            if self.wtmap is not None:
                return self.wtmap
            elif self.noisemap is not None:
                return 1/(self.noisemap)**2.
        elif type == 'ellipmask':
            return self.ellipmask
        elif type == 'segmask':
            return self.segmask
        elif type == 'uberseg':
            return self.ubersegmask
        elif type == 'totalmask':
            if self.ellipmask is not None:
                if self.ubersegmask is not None:
                    return self.ellipmask*self.ubersegmask
                elif self.segmask is not None:
                    return self.ellipmask*self.segmask
            else:
                if self.ubersegmask is not None:
                    return self.ubersegmask
                elif self.segmask is not None:
                    return self.segmask 
        elif type == 'wt_totalmask':
            if self.wtmap is not None:
                wt = self.wtmap
            elif self.noisemap is not None:
                wt = 1/(self.noisemap)**2.
            if self.ellipmask is not None:
                if self.ubersegmask is not None:
                    return wt*self.ellipmask*self.ubersegmask
                elif self.segmask is not None:
                    return wt*self.ellipmask*self.segmask
            else:
                if self.ubersegmask is not None:
                    return wt*self.ubersegmask
                elif self.segmask is not None:
                    return wt*self.segmask
                else:
                    return wt
        elif type == 'psf':
            return self.psfmap
        elif type == 'bgmask':
            return self.bgmask

    def plot(self, type='data', show=False, save=True):
        """
         Plot individual maps. 
         .. We multiply the datamap by available masks for better visualization 
            (otherwise extraneous pixels have overpowering brightness).
        """
        if (type == 'data'):
            plt.imshow(self.datamap*self.getMap(type='totalmask'),cmap='gray',origin='lower')
            plt.title(self.getLaTeXName())
        elif (type == 'noise'):
            if self.noisemap is not None:
                plt.imshow(self.noisemap,cmap='gray',origin='lower')
                plt.title(self.getLaTeXName()+' '+r'${\rm noise~map}$')
        elif (type == 'wt'):
            if self.wtmap is not None:
                plt.imshow(self.wtmap,cmap='gray',origin='lower')
                plt.title(self.getLaTeXName()+' '+r'${\rm weight~map}$')
        elif (type == 'totalmask'):
            plt.imshow(self.getMap(type='totalmask'),cmap='gray',origin='lower')
            plt.title(self.getLaTeXName()+' '+r'${\rm mask~map}$')
        elif (type == 'psf'):
            if self.psfmap is not None:
                plt.imshow(self.psfmap,cmap='gray',origin='lower')
                plt.title(self.getLaTeXName()+' '+r'${\rm PSF~map}$')
        if save == True:
            plt.savefig(self.getName()+'_'+type+'.pdf', format='pdf')
        if show == True:
            plt.show()

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
            f0 = np.sum(self.datamap*self.getMap('totalmask'))
            centroid[0] = np.sum(self.datamap*x*self.getMap('totalmask'))/f0 
            centroid[1] = np.sum(self.datamap*y*self.getMap('totalmask'))/f0 
            dx = x-centroid[0]
            dy = y-centroid[1]
            for idx in range(2):
                order1[idx] = np.sum(self.datamap*pow(dx,1-idx)*pow(dy,idx)*self.getMap('totalmask'))/f0
            for idx in range(3):
                order2[idx] = np.sum(self.datamap*pow(dx,2-idx)*pow(dy,idx)*self.getMap('totalmask'))/f0
            for idx in range(4):
                order3[idx] = np.sum(self.datamap*pow(dx,3-idx)*pow(dy,idx)*self.getMap('totalmask'))/f0
            for idx in range(5):
                order4[idx] = np.sum(self.datamap*pow(dx,4-idx)*pow(dy,idx)*self.getMap('totalmask'))/f0
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
        elif Qijkl == 'x':
            return x
        elif Qijkl == 'y':
            return y
        elif Qijkl == 'xc':
            return centroid[0]
        elif Qijkl == 'yc':
            return centroid[1]
        elif Qijkl == 'Q1':
            return order1[0]
        elif Qijkl == 'Q2':
            return order1[1]
        elif Qijkl == 'Q11':
            return order2[0]
        elif Qijkl == 'Q12':
            return order2[1]
        elif Qijkl == 'Q22':
            return order2[2]
        elif Qijkl == 'Q111':
            return order3[0]
        elif Qijkl == 'Q112':
            return order3[1]
        elif Qijkl == 'Q122':
            return order3[2]
        elif Qijkl == 'Q222':
            return order3[3]
        elif Qijkl == 'Q1111':
            return order4[0]
        elif Qijkl == 'Q1112':
            return order4[1]
        elif Qijkl == 'Q1122':
            return order4[2]
        elif Qijkl == 'Q1222':
            return order4[3]
        elif Qijkl == 'Q2222':
            return order4[4]
        elif Qijkl == 'all':
            return f0, x, y, centroid, order1, order2, order3, order4
        elif Qijkl == 'x,y,centroid,order2':
            return x, y, centroid, order2
        elif Qijkl == 'centroid,order2':
            return centroid, order2
        elif Qijkl == 'order2':
            return order2
        elif Qijkl =='order2,order3,order4':
            return order2, order3, order4

    def generateEllipticalMask(self, subtractBackground=True):
        """
        Here we generate the elliptical mask.
        During this process, we also estimate:
          (i).  the background
          (ii). the noisemap, in the case that a noisemap is not already provided
                to Lenser and read in through lenser_fits.py.
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
        if self.noisemap is None and self.wtmap is None and self.wtubersegmask is None:
            if self.gain is None:
                id_bg = np.where(self.bgmask==1)
                noise1 = np.ma.std(self.datamap[id_bg])*np.ones(self.datamap.shape)
                noise2 = 0 #noise2 = np.sqrt(abs(self.datamap*self.segmask))
                self.noisemap = np.sqrt(noise1**2.+noise2**2.)
            else:
                counts = self.datamap*self.getMap('totalmask')
                sky = self.sky*self.segmask
                id_bg = np.where(self.bgmask==1)
                noise1 = np.ma.std(self.datamap[id_bg])*np.ones(self.datamap.shape)
                noise2 = np.sqrt(abs(counts+sky+bg)/self.gain)
                self.noisemap = np.sqrt(noise1**2.+noise2**2.)
                
        # Calculate the elliptical mask
        # .. nsig is a heuristical number
        nsig_list = np.array((2.5,2.75,3,3.25,3.5,3.75,4.0))
        for i in range(len(nsig_list)):
            nsig = nsig_list[i]
            
            if self.wtmap is not None:
                id=np.where(self.datamap*self.getMap('totalmask') > nsig*(1/(np.sqrt(abs(self.wtmap))))*self.getMap('totalmask'))
            elif self.noisemap is not None:
                id=np.where(self.datamap*self.getMap('totalmask') > nsig*self.noisemap*self.getMap('totalmask'))
            
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
                
            if len(id[0]>10):
                self.ellipmask = np.zeros(self.datamap.shape)
                self.ellipmask[id] = 1
                break
            else:
                continue
        if len(id[0]>10):
            self.ellipmask = np.zeros(self.datamap.shape)
            self.ellipmask[id] = 1
        else:
            self.ellipmask = np.ones(self.datamap.shape)


            
class MultiImage(object):

    """
    Multi-Image class:
    .. Creates a list that holdes multiple Image() instantiations for a single galaxy.
    .. For use in multi-band and/or multi-epoch fitting.  
    """

    def __init__(self, namelist=None, datalist=None,
                 noiselist=None, wtlist=None,
                 ellipmasklist=None, seglist=None,
                 uberseglist=None, wtuberseglist=None,
                 bgmasklist=None, objmasklist=None,
                 psflist=None,
                 generateEllipticalMask_bool=True,
                 subtractBackground_bool=True):

        self.namelist = namelist
        self.datalist = datalist

        try:
            Nonelist = [None for i in range(len(datalist))]
        except:
            Nonelist = None
        
        if noiselist == None:
            noiselist = Nonelist
        self.noiselist = noiselist
        
        if wtlist == None:
            wtlist = Nonelist
        self.wtlist = wtlist
        
        if ellipmasklist == None:
            ellipmasklist = Nonelist
        self.ellipmasklist = ellipmasklist
        
        if seglist == None:
            seglist = Nonelist
        self.seglist = seglist
        
        if uberseglist == None:
            uberseglist = Nonelist
        self.uberseglist = uberseglist
        
        if wtuberseglist == None:
            wtuberseglist = Nonelist
        self.wtuberseglist = wtuberseglist
        
        if bgmasklist == None:
            bgmasklist = Nonelist
        self.bgmasklist = bgmasklist
        
        if objmasklist == None:
            objmasklist = Nonelist
        self.objmasklist = objmasklist
        
        if psflist == None:
            psflist = Nonelist
        self.psflist = psflist

        try:
            self.generateEllipticalMask_bool = generateEllipticalMask_bool
            self.subtractBackground_bool = subtractBackground_bool

            # Get number of epochs (i.e. number of observations for a single galaxy, be them different
            #   exposures in a single band, or across multiple bands)
            self.N_epochs = len(datalist)

            # Generate list of Image() instances
            self.Imagelist = []
            self.generateImagelist()
        except:
            self.Imagelist = Nonelist

    def generateImagelist(self):
        """
        Loop over all available epochs and create Image() instance for each one.
        """
        for i in range(self.N_epochs):
            im = Image(name = self.namelist[i], datamap=self.datalist[i],
                       noisemap = self.noiselist[i], wtmap = self.wtlist[i],
                       ellipmask = self.ellipmasklist[i], segmask = self.seglist[i],
                       ubersegmask = self.uberseglist[i], wtubersegmask = self.wtuberseglist[i],
                       bgmask = self.bgmasklist[i], objmask = self.objmasklist[i],
                       psfmap = self.psflist[i])
            if self.generateEllipticalMask_bool == True:
                im.generateEllipticalMask(subtractBackground = self.subtractBackground_bool)
            self.Imagelist.append(im)
            
            

"""
Module: lenser_fits
.. synopsis: Prepares galaxy fits files for use in Lenser
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>
"""



import os
import numpy as np
from astropy.io import fits

class FITS(object):

    """
    FITS class:
    .. Takes as an input the path to a galaxy postage stamp (as well as galaxy name).
    .. At the path location, FITS also searches for additional SExtractor outputs,
       such as a noise or weight map and segmentation map.
    .. If no noise or weight map is provided, FITS will search for a pickle file at 
       the location '../*noise-info.pkl' that contains information in order to 
       calculate one. If this pickle file does not exist, then a noisemap is calculated 
       based on simple assumptions in the lenser_galaxy module.  
    .. FITS will take the SExtractor segmentation map and construct an ubersegmentation mask
       (a Bitmask where any pixel that is closer to another object than the galaxy = 0)
       [MNRAS 460, 2245 (2016)]
    .. FITS will also search for a psf file associated with a given galaxy
    """

    def __init__(self, path, name = None, band = None):

        self.path = path
        self.name = name
        self.band = band

        self.fits_read()

    def fits_read(self):

        """
        fits_read function:
        .. Reads in FITS file for galaxy postage stamp
        .. Searches for additional corresponding files
        """

        #Read in galaxy postage stamp
        data_file = fits.open(self.path)[0].data

        # Look for noisemap, segmentation map, and PSFmap.  Check to see if the galaxy 
        # postage stamp name contains a specific visual band.
        if self.band != None:
            band = self.band
        else:
            if '_u' in self.path.split('.fits')[0]:
                band='u'
            elif '_g' in self.path.split('.fits')[0]:
                band='g'
            elif '_r' in self.path.split('.fits')[0]:
                band='r'
            elif '_i' in self.path.split('.fits')[0]:
                band='i'
            elif '_z' in self.path.split('.fits')[0]:
                band='z'
            else:
                 band=None
        if band != None:
            rms = self.path.split('_r.fits')[0]+'_'+band+'_rms.fits'
            seg = self.path.split('_r.fits')[0]+'_'+band+'_seg.fits'
            psf = self.path.split('_r.fits')[0]+'_'+band+'_psf.fits'
        else:
            rms = self.path.split('.fits')[0]+'_rms.fits'
            seg = self.path.split('.fits')[0]+'_seg.fits'
            psf = self.path.split('.fits')[0]+'_psf.fits'

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
            try:
                seg_map = fits.open(seg)['SCI'].data
            except:
                seg_map = fits.open(seg)['PRIMARY'].data

            # Get the segmentation map values corresponding to each object
            seg_ids = np.unique(seg_map)
            # Get background ids
            bg_val = 0
            id_bg = np.where(seg_map == bg_val)
            # Get indices for the center of the stamp
            seg_xc = int(seg_map.shape[0]/2)
            seg_yc = int(seg_map.shape[1]/2)
            obj_val = seg_map[seg_xc][seg_yc]
            id_obj = np.where(seg_map == obj_val)

            # Object mask corresponds to the object pixels only
            obj_mask = seg_map*0
            obj_mask[id_obj] = 1

            # Background mask corresponds to the background pixels only
            bg_mask = seg_map*0
            bg_mask[id_bg] = 1

            # Define segmentation mask
            seg_mask = seg_map*0
            seg_mask[id_obj] = 1
            seg_mask[id_bg] = 1

            # If there are extraneous objects within the stamp, create an uberseg mask
            if len(seg_ids) > 2:
                # .. Initialize uberseg mask
                ubserseg_mask = seg_map*0
                # .. Get id for all extraneous object pixels
                id_extr = np.where((seg_map != bg_val) & (seg_map != obj_val))
                # .. Get x,y coordinates for all object pixels
                x_obj, y_obj = id_obj[0][:], id_obj[1][:]
                # .. Get x,y coordinates for all extraneous object pixels
                x_extr, y_extr = id_extr[0][:], id_extr[1][:]

                # .. Uberseg:
                for i in range(seg_map.shape[0]):
                    for j in range(seg_map.shape[1]):
                        # .. Get distance of pixel (i,j) from every pixel in the object
                        d_obj = np.sqrt((x_obj-i)**2.+(y_obj-j)**2.)              
                        # .. Get distance of pixel (i,j) from every pixel in every extraneous object
                        d_extr = np.sqrt((x_extr-i)**2.+(y_extr-j)**2.)    
                        # .. Get minimum distance in each array
                        d_obj_min = np.min(d_obj)
                        d_extr_min = np.min(d_extr)
                        # .. If pixel is closer to main object than extranous object, include in uberseg
                        if d_obj_min <= d_extr_min: 
                            ubserseg_mask[i][j] = 1
                        else:
                            ubserseg_mask[i][j] = 0
            else:
                ubserseg_mask = 1

            # Full segmentation mask
            seg_mask*=ubserseg_mask

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
            try:
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
            except:
                gain = None
                sky = None
        else:
            gain = None
            sky = None

        # PSFmap
        if os.path.exists(psf):
            psf_file = fits.open(psf)[0].data
        else:
            psf_file = None

        # Set global variables
        self.datamap = data_file
        self.noisemap = rms_file
        self.segmask = seg_mask
        self.psfmap = psf_file
        self.bgmask = bg_mask
        self.obj_mask = obj_mask
        self.gain = gain
        self.sky = sky

    def get_FITS(self, type = 'data'):
        """
        Return various maps
        """
        if type == 'data':
            return self.datamap
        elif type == 'noise':
            return self.noisemap
        elif type == 'segmask':
            return self.segmask
        elif type == 'psf':
            return self.psfmap
        elif type == 'bgmask':
            return self.bgmask
        elif type == 'objmask':
            return self.objmask
        elif type == 'gain':
            return self.gain
        elif typw == 'sky':
            return self.sky

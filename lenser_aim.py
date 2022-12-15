"""
Module: lenser_aim
.. synopsis: Minimizes the parameter space of lenser_galaxy
.. module author: Evan J. Arena <evan.james.arena@drexel.edu>
"""

from lenser_galaxy import *

import numpy as np
import scipy
from scipy import optimize
from scipy.special import gamma
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import numdifftools as nd

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable

import copy
import warnings

verbose=False

class aimModel(object):

    """
    aimModel class:
    .. Estimates lensing signals from real survey data or realistically simulated images. 
    .. The module forward-models second-order lensing effects via calling lenser_galaxy, convolves with
       a PSF, and minimizes a parameter space. Previous studies on flexion signals have made use of 
       several techniques, including: moment analysis oflight distribution, decomposing images 
       into "shaplet" basis sets, and exploring the local potential field through a forward-modeling, 
       parameterized ray tracing known as Analytic Image Modeling (AIM). Lenser is intended as a 
       hybrid approach, first using a moment analysis to localize a best-fit lensing model in parameter 
       space and then performing a local minimization on the model parameters in lenser_galaxy
    .. There are six galaxy shape parameters and (up to) seven lens parameters:
    .. .. p = {xc,yc,ns,rs,q,phi,psi11,psi12,psi22,psi111,psi112,psi122,psi222}
    .. Recognizing the existence of the shear/ellipticity degeneracy, the default setting of lenser_aim
        is to set shear to zero (psi,ij = 0) and absorb the degenerate parameters into the intrinsic 
        ellipticity described by q and phi. In the context of smoothed mass-mapping, the inferred shear
        can be used as a prior.  This leaves us with a ten-parameter space given by
    .. .. p = {xc,yc,ns,rs,q,phi,psi111,psi112,psi122,psi222}
    .. lenser_aim estimates brightness moments from an unweighted quadrupole and hexadecapole 
       calculation to be used as an initial guess for the galaxy model. With initialized parameter 
       estimates provided by the measured light moments, the final stage of the lenser_aim pipeline 
       employs a two-step chisquared minimization: (i) first minimizing over the initially coupled 
       sub-space {ns, rs} (ii) a final full ten-parameter space local minimization
    """

    def __init__(self, myImage = None, myMultiImage = None, myGalaxy = Galaxy(), myLens = Lens(),
                 doFlags = np.array([1,1,1,1,1,1,0,0,0,1,1,1,1])):
        self.myGalaxy = myGalaxy
        self.myLens = myLens
        self.myGalaxy.setLens(myLens)
        self.I0 = 1.
        if myImage is not None:
            self.myImage = myImage
            self.mode = 'single'
        elif myMultiImage is not None:
            self.myMultiImage = myMultiImage
            self.mode = 'multi'

        # doFlags: Boolean describing which parameters should be fit
        # .. default: do not fit shear
        self.doFlags = doFlags
            
    def setPsi2(self, psi2new):
        """
        Set the values for the psi2 lensing array
        .. These include the covergence and shear terms
        .. psi2 = [psi11, psi12, psi22]
        """
        self.myGalaxy.setPsi2(psi2new)

    def setPsi3(self, psi3new):
        """
        Set the values for the psi3 lensing array
        .. These include the flexion terms
        .. psi3 = [psi111, psi112, psi122, psi222]
        """
        self.myGalaxy.setPsi3(psi3new)

    def setGalaxyPar(self, val, type):
        """ 
        Set the value for each galaxy shape parameter
        .. xc: centroid x coordinate
        .. yc: centroid y coordinate
        .. ns: seric index
        .. rs: characteristic radius
        .. q: semi-major-to-semi-minor axis ratio
        .. phi: orientation angle
        """
        self.myGalaxy.setPar(val, type)

    def parsWrapper(self):
        """
        Wrapper for the full parameter space of galaxy and lensing parameters.  
        .. Returns a numpy array.
        """
        xc = self.myGalaxy.xc
        yc = self.myGalaxy.yc
        ns = self.myGalaxy.ns
        rs = self.myGalaxy.rs
        q = self.myGalaxy.q
        phi = self.myGalaxy.phi
        psi2 = self.myLens.psi2
        psi3 = self.myLens.psi3
        pars = np.asarray([xc,yc,ns,rs,q,phi,
                           psi2[0],psi2[1],psi2[2],
                           psi3[0],psi3[1],psi3[2],psi3[3]])
        return pars

    def parsErrorWrapper(self):
        """
        Wrapper for the full parameter space of galaxy and lensing parameter errors.
        .. Returns a numpy array.
        """
        err = self.parsErrors
        return err

    def setPars(self, pars):
        """
        Set all of the galaxy and lensing parameters.
        .. Takes a numpy array of all parameters as the input, then sets the parameters using the 
           set_() functions which point to the Galaxy() class in the lenser_galaxy module.
        """
        xc = pars[0]
        yc = pars[1]
        ns = pars[2]
        rs = pars[3]
        q = pars[4]
        phi = pars[5]
        psi2 = pars[6:9]
        psi3 = pars[9:13]
        self.setGalaxyPar(xc,'xc')
        self.setGalaxyPar(yc,'yc')
        self.setGalaxyPar(ns,'ns')
        self.setGalaxyPar(rs,'rs')
        self.setGalaxyPar(q,'q')
        self.setGalaxyPar(phi,'phi')
        self.setPsi2(psi2)
        self.setPsi3(psi3)
        
    def calculateI0(self, image):
        """
        Calculate the central brightness of the modified S\'ersic proilfe, I0.
        """
        # Get the psf-convolved model image
        testImage = self.myGalaxy.generateImage(image.nx, image.ny, lens=True, psfmap=image.getMap(type='psf'))         
        I0 = np.sum(testImage.getMap(type='data')*image.getMap(type='data')*image.getMap(type='wt_totalmask'))\
            /np.sum(testImage.getMap(type='data')**2*image.getMap(type='wt_totalmask'))
        return I0

    def calculateI0Multi(self, multi_image):
        """
        Calculate the central brightness of the modified S\'ersic proilfe, I0, for each epoch in multi-fit.
        """
        I0_list = np.ones(multi_image.N_epochs)
        for i in range(multi_image.N_epochs):
            I0_list[i] = self.calculateI0(multi_image.Imagelist[i])
        return I0_list
                
    def setI0(self, I0):
        """
        Set the central brightness of the modified S\'ersic proilfe, I0.
        """
        self.I0 = I0   

    def chisq(self, image, reduced=True): 
        """
        Compute the chisquared: 
        .. chisqr = ((I_{model}(i, p) − I_{data}(i))^2)*w(i)
        .. I_{data} = self.myImage.getMap() 
        .. I_{model} = testImage.getMap()
        """
        # Get model image
        testImage = self.myGalaxy.generateImage(image.nx, image.ny, lens=True, I0=self.calculateI0(image),
                                                psfmap=image.getMap(type='psf'))
        # Calculate chisquared 
        chi2 = np.sum((testImage.getMap(type='data')-image.getMap(type='data'))**2.*image.getMap(type='wt_totalmask'))             
        if reduced == True:
            # Calculate reduced chisquared
            dof = np.sum(image.getMap(type='totalmask'))-np.sum(self.doFlags)
            chi2_reduced = chi2/dof
            return chi2_reduced
        elif reduced == False:
            return chi2

    def chisqMulti(self, multi_image, reduced=True):
        """
        Compute the chisquared for multi-fit.  This is simply a sum over the chisquared
        associated with fitting the model to each epoch.
        """
        chi2_tot = 0
        N_pix_tot = 0
        for i in range(len(multi_image.Imagelist)):
            chi2 = self.chisq(multi_image.Imagelist[i], reduced=False)
            if (np.isnan(chi2) == True):
                chi2=1e30
            chi2_tot += chi2
            N_pix_tot += np.sum(multi_image.Imagelist[i].getMap(type='totalmask'))
        if reduced == True:
            # Calculate reduced chisquared
            dof = N_pix_tot-np.sum(self.doFlags)
            chi2_tot_reduced = chi2_tot/dof
            return chi2_tot_reduced
        elif reduced == False:
            return chi2_tot

    def chisqWrapper(self, parsCurrent, parsFixed, doFlags):
        """
        Wrapper for the chisquared function.  
        .. Used for the local minimizer function.
        """
        pars = np.asarray(parsFixed)
        pars[np.where(doFlags==1)] = np.asarray(parsCurrent)
        self.setPars(pars)
        if self.mode == 'single':
            #return self.chisq(self.myImage)
            return self.chisq(self.myImage, reduced=False)
        elif self.mode == 'multi':
            return self.chisqMulti(self.myMultiImage)
  
    def size(self):
        """
        Return the characteristic size for a galaxy
          a = sqrt(Q11+Q22)
        """
        quadrupole = self.myGalaxy.GalaxyQuadrupole
        Q11 = quadrupole[0]
        Q22 = quadrupole[2]
        a = np.sqrt(abs(Q11+Q22))
        return a

    def rs_Q(self, ns):
        """
        Uses an analytic expression that relates ns and rs to Q11, Q22, and q.
        .. ns and rs are coupled, so this function takes ns as an input and returns rs.
        """
        quadrupole = self.myGalaxy.GalaxyQuadrupole
        Q11 = quadrupole[0]
        Q22 = quadrupole[2]
        Q0 = np.sqrt(abs(Q11+Q22))
        q = self.myGalaxy.q
        Q = Q0/np.sqrt(((1+q**2.)/2))
        rs = Q*np.sqrt(gamma(2.*ns)/gamma(4.*ns))
        return rs

    def flexionMoments(self, image, approximate=False):
        """
        Compute the first and second flexion from the image moments.
        """
        # Get the multipole image moments
        order2, order3, order4 = image.getMoments('order2,order3,order4')
        quadrupole = order2
        octupole = order3
        hexadecapole = order4

        Q11 = quadrupole[0]
        Q12 = quadrupole[1]
        Q22 = quadrupole[2]
        Q111 = octupole[0]
        Q112 = octupole[1]
        Q122 = octupole[2]
        Q222 = octupole[3]
        Q1111 = hexadecapole[0]
        Q1112 = hexadecapole[1]
        Q1122 = hexadecapole[2]
        Q1222 = hexadecapole[3]
        Q2222 = hexadecapole[4]

        # Define the Flexion estimators:
        F = np.zeros(2)
        G = np.zeros(2)

        # Define the Third-order HOLICs:
        #gamma = np.zeros(2)
        zeta = np.zeros(2)
        delta = np.zeros(2)

        # Useful definitions
        eta = np.zeros(2)
        lambd = np.zeros(2)
        xi = Q1111 + 2.*Q1122 + Q2222

        #; Compute gam
        #denom=Q11+Q22+2*sqrt(Q11*Q22-Q12^2)
        #gam[0]=(Q11-Q22)/denom
        #gam[1]=(2*Q12)/denom

        zeta[0] = (Q111 + Q122)/xi
        zeta[1] = (Q112 + Q222)/xi

        delta[0] = (Q111 - 3.*Q122)/xi
        delta[1] = (3.*Q112 - Q222)/xi

        if approximate==False:

            # Define the matrix M in 
            #   Mx = y
            # where x is a vector of the desired flexion estimators:
            #   x = (F1, F2, G1, G2)
            # and y is the measure of the third-order HOLICs:
            #   y = (xi1, xi2, delta1, delta2)
            M = np.zeros((4, 4))
            y = np.zeros(4)
            x = np.zeros(4)

            y[0] = zeta[0]
            y[1] = zeta[1]
            y[2] = delta[0]
            y[3] = delta[1]

            eta[0] = (Q1111 - Q2222)/xi
            eta[1] = 2.*(Q1112 + Q1222)/xi

            lambd[0] = (Q1111 - 6.*Q1122 + Q2222)/xi
            lambd[1] = 4.*(Q1112 - Q1222)/xi

            M[0][0] = 0.25*(9. + 8*eta[0])
            M[0][0] -= (33.*Q11**2. + 14.*Q11*Q22+Q22**2. + 20.*Q12**2.)/(4.*xi)

            M[0][1] = 0.25*(8.*eta[1])
            M[0][1] -= (32.*Q12*Q22 + 32.*Q11*Q12)/(4.*xi)

            M[0][2] = 0.25*(2*eta[0] + lambd[0])
            M[0][2] -= (3.*Q11**2. - 2.*Q11*Q22 - Q22**2. - 4.*Q12**2.)/(4.*xi)

            M[0][3] = 0.25*(2.*eta[1] + lambd[1])
            M[0][3] -= (2.*Q11*Q12)/xi

            M[1][0] = 0.25*(8*eta[1])
            M[1][0] -= (32.*Q12*Q22 + 32.*Q11*Q12)/(4.*xi)

            M[1][1] = 0.25*(-8*eta[0]+9.)
            M[1][1] -= (Q11**2. + 14.*Q11*Q22 + 20.*Q12**2. + 33.*Q22**2.)/(4.*xi)

            M[1][2] = 0.25*(-2*eta[1] + lambd[1])
            M[1][2] -= (-2*Q12*Q22)/xi

            M[1][3] = 0.25*(2.*eta[0] - lambd[0])
            M[1][3] -= (Q11**2. + 4.*Q12**2. + Q11*Q22 - 3.*Q22**2.)/(4.*xi)

            M[2][0] = 0.25*(10.*eta[0] + 7.*lambd[0])
            M[2][0] -= 3.*(11.*Q11**2. - 10.*Q11*Q22-Q22**2. - 20.*Q12**2.)/(4.*xi)

            M[2][1] = 0.25*(-10.*eta[1] + 7.*lambd[1])
            M[2][1] -= 3.*(8.*Q11*Q12 - 32.*Q12*Q22)/(4.*xi)

            M[2][2] = 0.25*(3.)
            M[2][2] -= 3.*(-2.*Q11*Q22 + Q11**2. + Q22**2. + 4.*Q12**2.)/(4.*xi)

            M[2][3] = 0.
            M[2][3] -= 0.

            M[3][0] = 0.25*(10.*eta[1] + 7.*lambd[1])
            M[3][0] -= 3.*(32.*Q11*Q12 - 8.*Q12*Q22)/(4.*xi)

            M[3][1] = 0.25*(10.*eta[0] - 7.*lambd[0])
            M[3][1] -= 3.*(Q11**2. + 20.*Q12**2. + 10.*Q11*Q22 - 11.*Q22**2.)/(4.*xi)

            M[3][2] = 0.
            M[3][2] -= 0.

            M[3][3] = 0.25*(3.)
            M[3][3] -= 3.*(-2.*Q11*Q22 + Q11**2. + Q22**2. + 4.*Q12**2.)/(4.*xi)

            M_inv = np.linalg.inv(M)
            x = np.matmul(M_inv, y)

            F[0] = x[0]
            F[1] = x[1]
            G[0] = x[2]
            G[1] = x[3]

        if approximate == True:
            # Compute approximate form of flexions
            F[0] = 4.*(Q111 + Q122)/(9.*xi - 12.*mu*xi)
            F[1] = 4.*(Q112 + Q222)/(9.*xi - 12.*mu*xi)
            G[0] = (4./3.)*delta[0]
            G[1] = (4./3.)*delta[1]

        return F, G

    def flexionToPsi3(self, F, G):
        """
        Convert the first and second flexion into the psi3 array.
        .. Takes as an input F and G, which are 1x2 numpy arrays of the form:
           F = [F1, F2], G = [G1, G2]
        """
        
        # Get F1, F2, G1, G2
        F1 = F[0]
        F2 = F[1]
        G1 = G[0]
        G2 = G[1]

        psi111 = (1./2.)*(3.*F1 + G1)
        psi112 = (1./2.)*(F2 + G2)
        psi122 = (1./2.)*(F1 - G1)
        psi222 = (1./2.)*(3.*F2 - G2)

        psi3 = np.zeros(4)
        psi3[0] = psi111
        psi3[1] = psi112
        psi3[2] = psi122
        psi3[3] = psi222

        return psi3

    def psi3ToFlexion(self, psi3 = 'useCurrentPsi3'):
        """
        Convert the psi3 array into the first and second flexion.
        .. Takes as an input psi3, which is a 1x4 array of the form:
           psi3 = [psi111, psi112, psi122, psi222]
        """

        # Unless otherwise specified, assumes that you want to convert the
        # currently set psi3 values.
        if psi3 == 'useCurrentPsi3':
            psi3 = self.myLens.psi3
        
        psi111 = psi3[0]
        psi112 = psi3[1]
        psi122 = psi3[2]
        psi222 = psi3[3]

        F = np.zeros(2)
        G = np.zeros(2)

        F[0] = (1./2.)*(psi111 + psi122)
        F[1] = (1./2.)*(psi112 + psi222)
        G[0] = (1./2.)*(psi111 - 3.*psi122)
        G[1] = (1./2.)*(3.*psi112 - psi222)

        return F, G
        
    def kappaAndGammaToPsi2(self, kappa, gamma):
        """
        Convert the convergence and shear into the psi2 array.
        .. Takes as an input kappa and gamma.  
        .. Kappa is a float and gamma is a 1x2 numpy array of the form:
             gamma = [gamma1, gamma2]
        """
        gamma1 = gamma[0]
        gamma2 = gamma[1]

        psi11 = kappa + gamma1
        psi12 = gamma2
        psi22 = kappa - gamma1

        psi2 = np.zeros(3)
        psi2[0] = psi11
        psi2[1] = psi12
        psi2[2] = psi22

        return psi2

    def simpleStart(self):
        """
        Estimates brightness moments from an unweighted quadrupole and hexadecapole 
        calculation to be used as an initial guess for the galaxy model.
        .. Estimates for {xc,yc,q,phi,psi111,psi112,psi122,psi222} are obtained
        """
        if self.mode == 'single':
            # Get image moments and centroid
            # .. Redefine the coordinates so that centroid is at the center of the stamp
            centroid, order2 = self.myImage.getMoments('centroid,order2')
            xc = centroid[0]-self.myImage.nx/2 
            yc = centroid[1]-self.myImage.ny/2
            Q11 = order2[0]
            Q12 = order2[1]
            Q22 = order2[2]
            
            # Calculate complex ellipticity terms to get phi and q
            chi1 = (Q11-Q22)/(Q11+Q22)
            chi2 = 2*Q12/(Q11+Q22)
            chisq = chi1**2+chi2**2
            phi = np.arctan2(chi2,chi1)/2.
            q = np.sqrt((1+np.sqrt(chisq))/(1-np.sqrt(chisq)))
            
            # Calculate HOLICs to get flexion terms
            F, G = self.flexionMoments(self.myImage)
            psi3 = self.flexionToPsi3(F, G)
            
        elif self.mode == 'multi':
            N_epochs = self.myMultiImage.N_epochs
            xc_list = np.empty(N_epochs)
            yc_list = np.empty(N_epochs)
            Q11_list = np.empty(N_epochs)
            Q12_list = np.empty(N_epochs)
            Q22_list = np.empty(N_epochs)
            q_list = np.empty(N_epochs)
            phi_list = np.empty(N_epochs)
            F1_list = np.empty(N_epochs)
            F2_list = np.empty(N_epochs)
            G1_list = np.empty(N_epochs)
            G2_list = np.empty(N_epochs)

            for i in range(N_epochs):
                # Get image moments and centroid
                # .. Redefine the coordinates so that centroid is at the center of the stamp
                centroid, order2 = self.myMultiImage.Imagelist[i].getMoments('centroid,order2')
                xc_list[i] = centroid[0]-self.myMultiImage.Imagelist[i].nx/2
                yc_list[i] = centroid[1]-self.myMultiImage.Imagelist[i].ny/2
                Q11 = order2[0]
                Q12 = order2[1]
                Q22 = order2[2]
                Q11_list[i] = Q11
                Q12_list[i] = Q12
                Q22_list[i] = Q22

                # Calculate complex ellipticity terms to get phi and q
                chi1 = (Q11-Q22)/(Q11+Q22)
                chi2 = 2*Q12/(Q11+Q22)
                chisq = chi1**2+chi2**2
                phi_list[i] = np.arctan2(chi2,chi1)/2.
                q_list[i] = np.sqrt((1+np.sqrt(chisq))/(1-np.sqrt(chisq)))
                
                # Calculate HOLICs to get flexion terms
                F, G = self.flexionMoments(self.myMultiImage.Imagelist[i])
                F1_list[i] = F[0]
                F2_list[i] = F[1]
                G1_list[i] = G[0]
                G2_list[i] = G[1]

            # Remove any epochs that return NaN
            # .. easiest to just add arrays by column (each column is an epoch) add see if
            #    any values add to nan
            id = np.where(np.isnan(xc_list+yc_list+Q11_list+Q12_list+Q22_list+q_list+phi_list\
                                   +F1_list+F2_list+G1_list+G2_list) == False)
            xc_list = xc_list[id]
            yc_list = yc_list[id]
            Q11_list = Q11_list[id]
            Q12_list = Q12_list[id]
            Q22_list = Q22_list[id]
            q_list = q_list[id]
            phi_list = phi_list[id]
            F1_list = F1_list[id]
            F2_list = F2_list[id]
            G1_list = G1_list[id]
            G2_list = G2_list[id]

            # Get average parameter values across all epochs and bands.
            #  For now, simply take this to be the median
            # .. May change to a weighted average based on each epoch's S/N in the future
            xc = np.median(xc_list)
            yc = np.median(yc_list)
            Q11 = np.median(Q11_list)
            Q12 = np.median(Q12_list)
            Q22 = np.median(Q22_list)
            q = np.median(q_list)
            phi = np.median(phi_list)
            F1 = np.median(F1_list)
            F2 = np.median(F2_list)
            G1 = np.median(G1_list)
            G2 = np.median(G2_list)
            F = np.array((F1,F2))
            G = np.array((G1,G2))
            psi3 = self.flexionToPsi3(F, G)
            
        # Set parameters
        # .. Set quadrupole moments for ease of access
        self.myGalaxy.setQuadrupole(np.array((Q11,Q12,Q22)))
        # .. Set shape parameters
        self.setGalaxyPar(xc,'xc')
        self.setGalaxyPar(yc,'yc')
        self.setGalaxyPar(q,'q')
        self.setGalaxyPar(phi,'phi')
        # .. Set lensing parameters
        self.setPsi3(psi3)

        # Calculate chisquared
        if self.mode == 'single':
            chi2 = self.chisq(self.myImage)
        elif self.mode == 'multi':
            chi2 = self.chisqMulti(self.myMultiImage)

        print('Parameter subspace initial guess from image moments:')
        print('...','[','xc =',xc,']')
        print('...','[','yc =',yc,']')
        print('...','[','q =',q,']')
        print('...','[','phi =',phi,']')
        print('...','[','psi,111 =',psi3[0],']')
        print('...','[','psi,112 =',psi3[1],']')
        print('...','[','psi,122 =',psi3[2],']')
        print('...','[','psi,222 =',psi3[3],']')
        print('...','[','Chisqr =',chi2,']')

    def localMin(self, bruteMin=False):
        """
        Performs a local minimzation over (subsets of) the parameter space
        .. doFlags represent which parameters should actually be included in a given minimzation run.
           Each element in the doFlags array corresponds to the parameter space given by
             p = {xc,yc,ns,rs,q,phi,psi11,psi12,psi22,psi111,psi112,psi122,psi222}
           A parameter is included in the minimzation if it has a correponding doFlag element equal to 1,
           and it is not included in the minimzation if it has a correponding doFlag element equal to 0.
        .. .. For example, doFlags = np.ones(13), means run a local minimzation over the entire parameter space
              wheras doFlags = [1,1,1,1,1,1,0,0,0,1,1,1,1] means run a local minimzation over a subset of the parameter
              space that does not include the psi2 terms.  The latter case is the default setting in this function, 
              as motivated by the shear/ellipticity degeneracy.
        .. There are two types of minimization routines built into this function:
        .. .. The brute-force routine (not the default routine chosen when localMin() is called)
        .. .. .. The purpose of this routine is specifically to minimze of an initially coupled subset of the parameter
                 space, {ns, rs}.  This routine creates a linspace of values for ns between 0.5 and 10 (i.e. the typical
                 range of values that the S\'ersic index takes), uses the function rs_Q() to return the corresponding 
                 rs value, and calculates the chisquared at each iteration.  The best-fit {ns, rs} corresponding to the 
                 minimum chisquared are then returned.  After this brute-force routine is complete, one should proceed to
                 run the next routine ...
        .. .. The L-BFGS-B local minimzation:
        .. .. .. An optimization algorithm in the family of quasi-Newton methods that approximates the 
                 Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS) using a limited amount of computer memory. 
        .. .. .. Included in the scipy optimization package
        .. .. .. L-BFGS-B is chosen specifically because it allows the inclusion of contraints on the parmeter space, 
                 which helps reduce run time as well as preventing an unrealistic minimization of model parameters.
        """
        pars0 = copy.deepcopy(self.parsWrapper())
        
        if  bruteMin == True:
            ns = np.linspace(0.1, 10., 100)
            rs = np.zeros((100))
            chisqrd = np.zeros((len(ns)))
            ibest = -1
            chimin = 1e9
            if self.mode == 'single':
                I0_best = 1.
            elif self.mode == 'multi':
                I0_best_list = np.ones(self.myMultiImage.N_epochs)

            for i in range(len(ns)):
                # Get parameter space
                parsCurrent = copy.deepcopy(self.parsWrapper())
                # Set ns and rs
                rs[i] = self.rs_Q(ns[i])
                parsCurrent[2] = ns[i]
                parsCurrent[3] = rs[i]
                # Chisquared minimization
                self.setPars(parsCurrent)
                if self.mode == 'single':
                    chival = self.chisq(self.myImage)
                    if verbose:print('chival', chival)
                    if (chival < chimin):
                        ibest = i
                        chimin = chival
                        I0_best = self.calculateI0(self.myImage)
                        if verbose:print(chimin)
                elif self.mode == 'multi':
                    chival = self.chisqMulti(self.myMultiImage)
                    if verbose:print('chival', chival)
                    if (chival < chimin):
                    #if (chival < chimin) & (chival > 0):
                        ibest = i
                        chimin = chival
                        I0_best = self.calculateI0Multi(self.myMultiImage)
                        if verbose:print(chimin)                    
            self.setGalaxyPar(ns[ibest],'ns')
            self.setGalaxyPar(rs[ibest],'rs')
            self.setI0(I0_best)

            # Calculate chisquared
            if self.mode == 'single':
                chi2 = self.chisq(self.myImage)
            elif self.mode == 'multi':
                chi2 = self.chisqMulti(self.myMultiImage)

            print('Brute force subspace minimization best-fit values:')
            print('...','[','ns =',ns[ibest],']')
            print('...','[','rs =',rs[ibest],']')
            print('...','[','Chisqr =',chi2,']')

        else:
            # Get (subset of) parameter space
            parsCurrent = copy.deepcopy(self.parsWrapper()[np.where(self.doFlags==1)]) 
            # Chisquared minimization
            out = optimize.minimize(self.chisqWrapper, parsCurrent, method='L-BFGS-B', args=(pars0,self.doFlags),
                                    bounds=((-1e3,1e3),(-1e3,1e3),(1e-1,5e1),(1e-10,1e2),(1,1e2),
                                            (None,None),(None,None),(None,None),(None,None),(None,None)),
                                    options={'disp':verbose,'maxiter':600})
            pars = np.asarray(pars0)
            pars[np.where(self.doFlags==1)] = np.asarray(out.x)
            self.setPars(pars)

            #print(out.success)
            #print(out.message)
            #print(out.nit)
            #print(out.keys())
            
            # Get errors on parameters
            ftol = 2.220446049250313e-09
            tmp_i = np.zeros(len(out.x))
            pars_err_tmp = np.zeros(len(out.x))
            for i in range(len(out.x)):
                tmp_i[i] = 1.0
                hess_inv_i = out.hess_inv(tmp_i)[i]
                #pars_err_tmp[i] = np.sqrt(max(1, abs(out.fun)) * ftol * hess_inv_i)
                pars_err_tmp[i] = np.sqrt(abs(out.fun) * ftol * hess_inv_i)
                tmp_i[i] = 0.0
            pars_err = np.zeros(len(self.parsWrapper()))
            pars_err[np.where(self.doFlags==1)] = pars_err_tmp
            # .. Set errors
            self.parsErrors = pars_err
            
            # Calculate I0
            if self.mode == 'single':   
                I0 = self.calculateI0(self.myImage)
            elif self.mode == 'multi':
                I0 = self.calculateI0Multi(self.myMultiImage)
            # .. Set I0
            self.setI0(I0)

            # Calculate chisquared
            if self.mode == 'single':
                chi2 = self.chisq(self.myImage)
            elif self.mode == 'multi':
                chi2 = self.chisqMulti(self.myMultiImage)
            # .. Set chisquared
            self.chisquared = chi2
            
            # Print bestfit parameters
            print('L-BFGS-B local minimzation best-fit values:')
            print('...','[','xc =',pars[0],']')
            print('...','[','yc =',pars[1],']')
            print('...','[','ns =',pars[2],']')
            print('...','[','rs =',pars[3],']')  
            print('...','[','q =',pars[4],']')
            print('...','[','phi =',pars[5],']') 
            print('...','[','psi,11 =',pars[6],']')
            print('...','[','psi,12 =',pars[7],']')
            print('...','[','psi,22 =',pars[8],']')
            print('...','[','psi,111 =',pars[9],']')
            print('...','[','psi,112 =',pars[10],']')
            print('...','[','psi,122 =',pars[11],']')
            print('...','[','psi,222 =',pars[12],']')
            print('...','[','Chisqr =',chi2,']')

            # Print 1sigma errors
            print('1sigma errors on parameters:')
            print('...','[','error on xc =',pars_err[0],']')
            print('...','[','error on yc =',pars_err[1],']')
            print('...','[','error on ns =',pars_err[2],']')
            print('...','[','error on rs =',pars_err[3],']')  
            print('...','[','error on q =',pars_err[4],']')
            print('...','[','error on phi =',pars_err[5],']') 
            print('...','[','error on psi,11 =',pars_err[6],']')
            print('...','[','error on psi,12 =',pars_err[7],']')
            print('...','[','error on psi,22 =',pars_err[8],']')
            print('...','[','error on psi,111 =',pars_err[9],']')
            print('...','[','error on psi,112 =',pars_err[10],']')
            print('...','[','error on psi,122 =',pars_err[11],']')
            print('...','[','error on psi,222 =',pars_err[12],']')    

    def globalMin(self):
        """
        We include the option for global minimization rather than a two-step local minimzation.
        This method is not necessarily recommended as superior, as the much larger computation time
        does not seem to provide much in the way of an improvement over the accuracy of the local
        minimzation routine localMin().
        .. Basin-hopping is a global optimization technique that iterates by performing random perturbation 
           of coordinates, performing local optimization, and accepting or rejecting new coordinates based 
           on a minimized function value.
        .. Included in the scipy optimization package.
        .. NOTE: This works for single-fit mode only.
        """
        pars0 = copy.deepcopy(self.parsWrapper())
        # Get (subset of) parameter space
        parsCurrent = copy.deepcopy(self.parsWrapper()[np.where(self.doFlags==1)]) 
        # Chisquared minimization
        out = optimize.basinhopping(self.chisqWrapper, parsCurrent, minimizer_kwargs={'method':'L-BFGS-B','args':(pars0,self.doFlags)})
        pars = np.asarray(pars0)
        pars[np.where(self.doFlags==1)] = np.asarray(out.x)
        self.setPars(pars)
        I0 = self.calculateI0()
        self.setI0(I0)
        print('Basinhopping global minimzation best-fit values:')
        print('...','[','xc =',pars[0],']')
        print('...','[','yc =',pars[1],']')
        print('...','[','ns =',pars[2],']')
        print('...','[','rs =',pars[3],']')  
        print('...','[','q =',pars[4],']')
        print('...','[','phi =',pars[5],']') 
        print('...','[','psi,11 =',pars[6],']')
        print('...','[','psi,12 =',pars[7],']')
        print('...','[','psi,22 =',pars[8],']')
        print('...','[','psi,111 =',pars[9],']')
        print('...','[','psi,112 =',pars[10],']')
        print('...','[','psi,122 =',pars[11],']')
        print('...','[','psi,222 =',pars[12],']')
        print('...','[','Chisqr =',self.chisq(),']')

    def runLocalMinRoutine(self):
        """
        First, call simpleStart() to estimate brightness moments from an unweighted quadrupole and hexadecapole 
        calculation to be used as an initial guess for the galaxy model. With initialized parameter 
        estimates provided by the measured light moments, perform a two-step chisquared minimization using localMin():
        (i) first minimizing over the initially coupled subspace {ns, rs} (ii) a final full ten-parameter space local 
        minimization (where doFlags are zero for the psi2 terms by default, due to the shear/ellipticity degeneracy.
        """
        self.simpleStart()
        self.localMin(bruteMin=True)
        self.localMin()
        
    def checkFit(self, image):
        """
        Check for a realistic fit.     
        .. Mostly checking for bad flexion values that create secondary lens images within the postage stamp.
        .. This is done by checking to see if there are two 'centroid regions.'  If there are, an error is raised.
        .. NOTE: this works for single-fit only
        """
        psf = image.getMap('psf')
        myModel = self.myGalaxy.generateImage(image.nx,image.ny,I0=self.I0,lens=True,psfmap=psf).getMap()

        neighborhood_size = 5
        threshold = 0.005*self.I0
        data = myModel
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(y_center)
        
        if num_objects == 1:
            print('Good fit')
            return 0
        else:
            print('Bad fit')
            return 1

    def make_plot_compare(self, show=False, save=True, zoom=False, bruteMin=False):
        """
        Create a three-panel plot of the original galaxy image, the model image, and the residual.
        .. We multiply the datamap by the segmask for plotting purposes only, 
           for better visualization (otherwise extraneous pixels have overpowering brightness).
        .. NOTE: this works for single-fit only
        """
        # Get original galaxy image
        myImage = self.myImage.getMap()*self.myImage.getMap(type='totalmask')
        # Get PSF
        psf = self.myImage.getMap('psf')
        # Generate model image and convolve with the PSF
        myModel = self.myGalaxy.generateImage(self.myImage.nx, self.myImage.ny, lens=True, I0=self.I0, psfmap=psf).getMap()
        # Get residual
        difference = myImage-myModel
        # Make plot
        f, ax = plt.subplots(1, 3)
        im1 = ax[0].imshow(myImage,cmap='gray',origin='lower',vmin=np.min(difference),vmax=np.max(myImage)) 
        im2 = ax[1].imshow(myModel,cmap='gray',origin='lower',vmin=np.min(difference),vmax=np.max(myImage))
        im3 = ax[2].imshow(difference,cmap='gray',origin='lower',vmin=np.min(difference),vmax=np.max(myImage))
        ax[0].set_title(self.myImage.getLaTeXName())
        ax[1].set_title(r'${\rm \texttt{Lenser} \, Model}$')
        ax[2].set_title(r'${\rm Residual}$')

        plt.tight_layout()            

        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        f.colorbar(im1, ax=ax[0], cax=cax1)
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        f.colorbar(im2, ax=ax[1], cax=cax2)
        divider3 = make_axes_locatable(ax[2])
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        f.colorbar(im3, ax=ax[2], cax=cax3)

        plt.tight_layout()    
        
        if zoom == False:
            if save == True:
                if bruteMin == True:
                    plt.savefig(self.myImage.getName()+'_bruteMin.pdf', format='pdf',bbox_inches='tight')
                elif bruteMin == False:
                    plt.savefig(self.myImage.getName()+'_localMin.pdf', format='pdf',bbox_inches='tight')
            if show == True:
                plt.show()
        elif zoom == True:
            A = np.sum(self.myImage.getMap(type='totalmask'))
            q = self.myGalaxy.q
            a = np.sqrt((A*q)/np.pi)
            buf = a
            xleft = (self.myImage.nx/2+self.myGalaxy.xc)-buf
            xright = (self.myImage.nx/2+self.myGalaxy.xc)+buf
            ybottom = (self.myImage.ny/2+self.myGalaxy.yc)-buf
            ytop = (self.myImage.ny/2+self.myGalaxy.yc)+buf
            ax[0].set_xlim(xleft,xright)
            ax[0].set_ylim(ybottom,ytop)
            ax[1].set_xlim(xleft,xright)
            ax[1].set_ylim(ybottom,ytop)
            ax[2].set_xlim(xleft,xright)
            ax[2].set_ylim(ybottom,ytop)
            if save == True:
                if bruteMin == True:
                    plt.savefig(self.myImage.getName()+'_bruteMin_zoom.pdf', format='pdf',bbox_inches='tight')
                elif bruteMin == False:
                    plt.savefig(self.myImage.getName()+'_localMin_zoom.pdf', format='pdf',bbox_inches='tight')
            if show == True:
                plt.show()

    def empty(self):
        """
        Reset the parameters to their default values
        .. This function is necessary if you are looping over multiple images
        """
        emptyGalaxy = Galaxy()
        emptyLens = Lens()
        xc = emptyGalaxy.xc
        yc = emptyGalaxy.yc
        ns = emptyGalaxy.ns
        rs = emptyGalaxy.rs
        q = emptyGalaxy.q
        phi = emptyGalaxy.phi
        psi2 = emptyLens.psi2
        psi3 = emptyLens.psi3
        pars = np.asarray([xc,yc,ns,rs,q,phi,
                        psi2[0],psi2[1],psi2[2],
                        psi3[0],psi3[1],psi3[2],psi3[3]])
        self.setPars(pars)

        self.I0=1.
        
        if self.mode == 'single':
            emptyImage = Image()
            self.myImage = emptyImage
        elif self.mode == 'multi':
            emptyMulti = MultiImage()
            self.myMultiImage = emptyMulti

        


        

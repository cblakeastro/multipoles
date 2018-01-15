########################################################################
# Code to convolve model power spectrum multipoles in Fourier bins     #
# with a gridded survey window function and weight function, using     #
# spherical harmonics.                                                 #
#                                                                      #
# The algorithm is based on Blake et al., 2018, MNRAS, submitted.      #
#                                                                      #
# Arguments:                                                           #
# nx,ny,nz -- size of gridded cuboid                                   #
# lx,ly,lz -- dimensions of gridded cuboid [Mpc/h]                     #
# x0,y0,z0 -- co-ordinate origin in cuboid co-ordinates [Mpc/h]        #
# kmin,kmax -- minimum,maximum wavenumber for binning [h/Mpc]          #
# nkbin -- number of k bins                                            #
# weigrid -- gridded weight function                                   #
# wingrid -- gridded window function                                   #
# kmod,pkmod -- power spectrum array [k in h/Mpc, P in (Mpc/h)^3]      #
# calcgrowth -- True if scaling a z=0 power spectrum using D(z)^2      #
# zeff -- effective redshift if calcgrowth=True                        #
# calcbeta -- True if determining beta as f/b in fiducial cosmology    #
# beta -- beta value if calcbeta=False                                 #
# sigopt -- pairwise velocity dispersion: (1) Lorentzian (2) Gaussian  #
# sigv -- value of pairwise velocity dispersion                        #
# b -- galaxy bias                                                     #
# cosmo -- astropy fiducial cosmology                                  #
#                                                                      #
# Returns:                                                             #
# pk0con,pk2con,pk4con --  convolved power spectrum multipoles         #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
from scipy.special import sph_harm
import getpolemod
import tools

# Convolve power spectrum multipoles using spherical harmonics
def getpoleconv(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,weigrid,wingrid,kmod,pkmod,calcgrowth,zeff,calcbeta,beta,sigopt,sigv,b,cosmo):
  print '\nConvolving multipole power spectra using spherical harmonics...'
  nl = 3    # Number of convolved multipoles to compute
  nlmod = 3 # Number of unconvolved multipoles in model
# Initializations
  nc = float(nx*ny*nz)
  dx,dy,dz = lx/nx,ly/ny,lz/nz
  x = dx*np.arange(nx) - x0
  y = dy*np.arange(ny) - y0
  z = dz*np.arange(nz) - z0
  x += 0.5*dx
  y += 0.5*dy
  z += 0.5*dz
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  kgrid = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  doindep,dohalf = False,False
  sumw = np.sum(wingrid)
  sumwsq = nc*np.sum(((wingrid/sumw)*weigrid)**2)
# Unconvolved power spectrum multipoles
  dk = (kmax-kmin)/nkbin
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  pk0mod,pk2mod,pk4mod = getpolemod.getpkpolemod(kmin,kmax,nkbin,kmod,pkmod,calcgrowth,zeff,calcbeta,beta,sigopt,sigv,b,cosmo)
# Obtain spherical polar angles over the grid
  xthetagrid = np.arctan2(z[np.newaxis,np.newaxis,:],y[np.newaxis,:,np.newaxis])
  xphigrid = np.where(rgrid>0.,np.arccos(x[:,np.newaxis,np.newaxis]/rgrid),0.)
  kthetagrid = np.arctan2(kz[np.newaxis,np.newaxis,:],ky[np.newaxis,:,np.newaxis])
  kphigrid = np.where(kgrid>0.,np.arccos(kx[:,np.newaxis,np.newaxis]/kgrid),0.)
# Fourier transform window function
  winspec = np.fft.fftn(weigrid*(wingrid/sumw))
# Compute convolutions
  pk0con,pk2con,pk4con = np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin)
  for il in range(nl):
    l = 2*il
    pkcongrid = np.zeros((nx,ny,nz))
    for m in range(-l,l+1):
      xylmgrid = sph_harm(m,l,xthetagrid,xphigrid)
      kylmgrid = sph_harm(m,l,kthetagrid,kphigrid)
      for ilp in range(nlmod):
        lp = 2*ilp
        norm = ((4.*np.pi)**2)/(sumwsq*float(2*lp+1))
        print 'Computing convolution for l =',l,'m =',m,'lp =',lp,'...'
        if (ilp == 0):
          plpgrid = np.interp(kgrid,kbin,pk0mod)
        elif (ilp == 1):
          plpgrid = np.interp(kgrid,kbin,pk2mod)
        elif (ilp == 2):
          plpgrid = np.interp(kgrid,kbin,pk4mod)
        for mp in range(-lp,lp+1):
          xylmpgrid = sph_harm(mp,lp,xthetagrid,xphigrid)
          kylmpgrid = sph_harm(mp,lp,kthetagrid,kphigrid)
          pkmodspec = np.fft.fftn(plpgrid*np.conj(kylmpgrid))
          slmlmpgrid = np.fft.fftn(weigrid*(wingrid/sumw)*xylmgrid*np.conj(xylmpgrid))
          tempspec = np.fft.fftn(winspec*np.conj(slmlmpgrid))
          pkcongrid += norm*np.real(kylmgrid*np.fft.ifftn(pkmodspec*tempspec))
# Average over k modes
    pkcon,nmodes = tools.binpk(pkcongrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
    if (il == 0):
      pk0con = pkcon
    elif (il == 1):
      pk2con = pkcon
    elif (il == 2):
      pk4con = pkcon
# Display measurements:
  print '\nConvolved model power spectra:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:9.1f}'.format(pk0con[ik]),'{:9.1f}'.format(pk2con[ik]),'{:9.1f}'.format(pk4con[ik])
  return pk0con,pk2con,pk4con

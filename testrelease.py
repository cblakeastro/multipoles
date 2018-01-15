########################################################################
# Example code for analysing 6dFGS power spectrum multipoles.          #
# The following functions are available:                               #
#                                                                      #
# getpoleest - estimate power spectrum multipoles                      #
# getpolemod - compute model power spectrum multipoles                 #
# getpoleconv - convolve model with survey window function             #
# getpolewin - determine window function multipoles                    #
# getpolecov - determine covariance in the Gaussian approximation      #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import sys
sys.path.insert(0,'/Users/cblake/Work/tools/coderelease/')
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import tools
import getpoleest
import getpoleconv
import getpolewin
import getpolecov
import getpolemod

def main():

  datfile = '6dfgs_data.dat' # galaxy co-ordinates
  winfile = 'winx_6dfgs_zmax0pt10.dat' # window function file
  pkmodfile = 'pkcambhalofit_zeq0_6dfgscola.dat' # power spectrum model
  kmin = 0.   # minimum wavenumber for binning [h/Mpc]
  kmax = 0.1  # maximum wavenumber for binning [h/Mpc]
  nkbin = 5   # number of k bins
  pk0 = 1600. # power spectrum amplitude for FKP weights [(Mpc/h)^3]
  nmax = 100  # number of modes to use in each shell for covariance estimate

# Model power spectrum parameters
  data = np.loadtxt(pkmodfile)
  kmod,pkmod = data[:,0],data[:,1]
  cosmo = FlatLambdaCDM(H0=100.,Om0=0.3)
  calcgrowth,calcbeta = True,False
  zeff = 0.   # model redshift
  sigopt = 1  # pairwise velocity dispersion: (1) Lorentzian (2) Gaussian
  sigv = 300. # value of pairwise velocity dispersion
  b = 1.45    # galaxy bias
  beta = (cosmo.Om(zeff)**0.55)/b # model beta value

# Read in 6dFGS data
  print '\nReading in data...'
  print datfile
  data = np.loadtxt(datfile)
  xpos,ypos,zpos = data[:,0],data[:,1],data[:,2]
  ngal = len(xpos)
  print 'Number of galaxies =',ngal
  
# Determine box size and origin in cuboid co-ordinates
  zmax = 0.1
  rmax = cosmo.comoving_distance(zmax).value
  lx,ly,lz = 2.*rmax,2.*rmax,rmax
  vol = lx*ly*lz
  x0,y0,z0 = rmax,rmax,rmax

# Read in 6dFGS window function
  wingrid,nx,ny,nz = tools.readwin(winfile)
  sumw = np.sum(wingrid)
  
# Determine gridded weight function
  nc = float(nx*ny*nz)
  weigrid = 1./(1.+(((wingrid/sumw)*ngal*pk0*nc)/vol))

# Grid data
  datgrid = tools.discret(xpos,ypos,zpos,nx,ny,nz,lx,ly,lz,x0,y0,z0)

# Measure multipole power spectra
  pk0,pk2,pk4,nmodes = getpoleest.getpoleest(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,datgrid,weigrid,wingrid)

# Obtain model power spectrum
  pk0mod,pk2mod,pk4mod = getpolemod.getpkpolemod(kmin,kmax,nkbin,kmod,pkmod,calcgrowth,zeff,calcbeta,beta,sigopt,sigv,b,cosmo)

# Convolve multipole power spectra
  pk0con,pk2con,pk4con = getpoleconv.getpoleconv(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,weigrid,wingrid,kmod,pkmod,calcgrowth,zeff,calcbeta,beta,sigopt,sigv,b,cosmo)

# Obtain window function multipoles
  smin,smax,nsbin = 0.,200.,41
  swin,w0sqwin,w2sqwin,w4sqwin,w6sqwin,w8sqwin = getpolewin.getpolewin(smin,smax,nsbin,nx,ny,nz,lx,ly,lz,x0,y0,z0,weigrid,wingrid)

# Determine power spectrum covariance
  pkcov = getpolecov.getpolecov(nx,ny,nz,lx,ly,lz,x0,y0,z0,ngal,kmin,kmax,nkbin,nmax,pk0con,pk2con,pk4con,weigrid,wingrid)

  return

if __name__ == '__main__':
  main()

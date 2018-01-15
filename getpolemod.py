########################################################################
# Code to determine unconvolved model power spectrum multipoles.       #
#                                                                      #
# Arguments:                                                           #
# kmin,kmax -- minimum,maximum wavenumber for binning [h/Mpc]          #
# nkbin -- number of k bins                                            #
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
# pk0mod,pk2mod,pk4mod --  model power spectrum multipoles             #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
from scipy.integrate import quad

# Obtain arrays of model power spectrum multipoles
def getpkpolemod(kmin,kmax,nkbin,kmod,pkmod,calcgrowth,zeff,calcbeta,beta,sigopt,sigv,b,cosmo):
  dk = (kmax-kmin)/nkbin
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  pk0mod,pk2mod,pk4mod = getpkpolersd(kbin,kmod,pkmod,calcgrowth,zeff,calcbeta,beta,sigopt,sigv,b,cosmo)
  print '\nModel power spectra:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:9.1f}'.format(pk0mod[ik]),'{:9.1f}'.format(pk2mod[ik]),'{:9.1f}'.format(pk4mod[ik])
  return pk0mod,pk2mod,pk4mod

# Obtain power spectrum multipoles at a given scale k
def getpkpolersd(k,kmod,pkmod,calcgrowth,zeff,calcbeta,beta,sigopt,sigv,b,cosmo):
  if (calcgrowth):
    dz = growth(zeff,cosmo)
  else:
    dz = 1.
  if (calcbeta):
    beta = (cosmo.Om(zeff)**0.55)/b
  rsd0fact = [ poleboost(0,k1,beta,sigopt,sigv) for k1 in k ]
  rsd2fact = [ poleboost(2,k1,beta,sigopt,sigv) for k1 in k ]
  rsd4fact = [ poleboost(4,k1,beta,sigopt,sigv) for k1 in k ]
  norm2 = np.full(len(k),5.)
  norm4 = np.full(len(k),9.)
  pk = np.interp(k,kmod,pkmod)*(b**2)*(dz**2)
  return rsd0fact*pk,norm2*rsd2fact*pk,norm4*rsd4fact*pk

# Perform integration over mu
def poleboost(l,k,beta,sigopt,sigv):
  poleboost,err = quad(polemuboost,0.,1.,args=(l,k,beta,sigopt,sigv))
  return poleboost

# Integrand as a function of mu
def polemuboost(mu,l,k,beta,sigopt,sigv):
  if (l == 2):
    leg = (3.*(mu**2)-1.)/2.
  elif (l == 4):
    leg = (35.*(mu**4)-30.*(mu**2)+3.)/8.
  else:
    leg = 1.
  return leg*pkmuboost(mu,k,beta,sigopt,sigv)

# Multiplicative term for power spectrum P(k,mu)
def pkmuboost(mu,k,beta,sigopt,sigv):
  nom = (1.+(beta*(mu**2)))**2
  if (sigopt == 1):
    den = 1. + ((k*0.01*sigv*mu)**2)
  else:
    den = np.exp((k*0.01*sigv*mu)**2)
  return nom/den

# Growth factor approximation
def growth(z,cosmo):
  om0 = cosmo.Om(0.)
  ol0 = cosmo.Ode(0.)
  omz = cosmo.Om(z)
  olz = cosmo.Ode(z)
  temp = (om0**(4./7.)) - ol0 + ((1.+(om0/2.))*(1.+(ol0/70.)))
  dz0 = (2.5*om0)/temp
  temp = (omz**(4./7.)) - olz + ((1.+(omz/2.))*(1.+(olz/70.)))
  dz = (2.5*omz)/(temp*(1.+z)*dz0)
  return dz

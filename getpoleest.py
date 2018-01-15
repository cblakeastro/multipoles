########################################################################
# Code to estimate the power spectrum multipoles in Fourier bins from  #
# a gridded galaxy distribution, window function and weight function.  #
#                                                                      #
# The algorithm is based on Bianchi et al., 2015, MNRAS, 453, L11.     #
#                                                                      #
# Arguments:                                                           #
# nx,ny,nz -- size of gridded cuboid                                   #
# lx,ly,lz -- dimensions of gridded cuboid [Mpc/h]                     #
# x0,y0,z0 -- co-ordinate origin in cuboid co-ordinates [Mpc/h]        #
# kmin,kmax -- minimum,maximum wavenumber for binning [h/Mpc]          #
# nkbin -- number of k bins                                            #
# datgrid -- gridded data                                              #
# weigrid -- gridded weight function                                   #
# wingrid -- gridded window function                                   #
#                                                                      #
# Returns:                                                             #
# pk0,pk2,pk4 -- power spectrum multipoles                             #
# nmodes -- number of modes in each bin                                #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
import tools

def getpoleest(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,datgrid,weigrid,wingrid):
  print '\nEstimating multipole power spectra with Bianchi et al. method...'
  # Initializations
  nc = float(nx*ny*nz)
  dx,dy,dz,vol = lx/nx,ly/ny,lz/nz,lx*ly*lz
  x = dx*np.arange(nx) - x0
  y = dy*np.arange(ny) - y0
  z = dz*np.arange(nz) - z0
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)[:nz/2+1]
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  rgrid[rgrid == 0.] = 1.
  kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  kspec[0,0,0] = 1.
  ngal = np.sum(datgrid)
  sumw = np.sum(wingrid)
  sumwsq = nc*np.sum(((wingrid/sumw)*weigrid)**2)
# Determine F(x)
  fgrid = weigrid*(datgrid - ngal*(wingrid/sumw))
# Determine shot noise factor Sum w(x)^2 N(x)
  sgal = np.sum((weigrid**2)*datgrid)
# Determine A_0(k)
  print 'Determining A_0(k)...'
  a0spec = np.fft.rfftn(fgrid)
# Determine A_2(k)
  print 'Determining A_2(k)...'
  a2spec = np.zeros((nx,ny,nz/2+1),dtype=complex)
  for iterm in range(1,7):
    rfactgrid,kfactspec = geta2term(iterm,x,y,z,kx,ky,kz)
    tempgrid = (rfactgrid*fgrid)/(rgrid**2)
    tempspec = np.fft.rfftn(tempgrid)
    a2spec += (kfactspec*tempspec)/(kspec**2)
# Determine A_4(k)
  print 'Determining A_4(k)...'
  a4spec = np.zeros((nx,ny,nz/2+1),dtype=complex)
  for iterm in range(1,16):
    rfactgrid,kfactspec = geta4term(iterm,x,y,z,kx,ky,kz)
    tempgrid = (rfactgrid*fgrid)/(rgrid**4)
    tempspec = np.fft.rfftn(tempgrid)
    a4spec += (kfactspec*tempspec)/(kspec**4)
# Power spectrum estimators
  pk0spec = (np.real(a0spec*np.conj(a0spec))-sgal)*vol/(sumwsq*(ngal**2))
  pk2spec = np.real(a0spec*np.conj(3.*a2spec-a0spec))*5.*vol/(2.*sumwsq*(ngal**2))
  pk4spec = np.real(a0spec*np.conj(35.*a4spec-30.*a2spec+3.*a0spec))*9.*vol/(8.*sumwsq*(ngal**2))
# Average power spectra in bins
  doindep,dohalf = True,True
  pk0,nmodes = tools.binpk(pk0spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk2,nmodes = tools.binpk(pk2spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk4,nmodes = tools.binpk(pk4spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
# Display measurements:
  dk = (kmax-kmin)/nkbin
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  print '\nPower spectrum measurements:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:9.1f}'.format(pk0[ik]),'{:9.1f}'.format(pk2[ik]),'{:9.1f}'.format(pk4[ik]),'{:6d}'.format(nmodes[ik])
  return pk0,pk2,pk4,nmodes

# A_2(k) defined by Equation 10 in Bianchi et al.
def geta2term(iterm,x,y,z,kx,ky,kz):
  if (iterm == 1):
    rfact = x[:,np.newaxis,np.newaxis]**2
    kfact = kx[:,np.newaxis,np.newaxis]**2
  elif (iterm == 2):
    rfact = y[np.newaxis,:,np.newaxis]**2
    kfact = ky[np.newaxis,:,np.newaxis]**2
  elif (iterm == 3):
    rfact = z[np.newaxis,np.newaxis,:]**2
    kfact = kz[np.newaxis,np.newaxis,:]**2
  elif (iterm == 4):
    rfact = x[:,np.newaxis,np.newaxis]*y[np.newaxis,:,np.newaxis]
    kfact = 2.*kx[:,np.newaxis,np.newaxis]*ky[np.newaxis,:,np.newaxis]
  elif (iterm == 5):
    rfact = x[:,np.newaxis,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 2.*kx[:,np.newaxis,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 6):
    rfact = y[np.newaxis,:,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 2.*ky[np.newaxis,:,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  return rfact,kfact

# A_4(k) defined by Equation 12 in Bianchi et al.
def geta4term(iterm,x,y,z,kx,ky,kz):
  if (iterm == 1):
    rfact = x[:,np.newaxis,np.newaxis]**4
    kfact = kx[:,np.newaxis,np.newaxis]**4
  elif (iterm == 2):
    rfact = y[np.newaxis,:,np.newaxis]**4
    kfact = ky[np.newaxis,:,np.newaxis]**4
  elif (iterm == 3):
    rfact = z[np.newaxis,np.newaxis,:]**4
    kfact = kz[np.newaxis,np.newaxis,:]**4
  elif (iterm == 4):
    rfact = (x[:,np.newaxis,np.newaxis]**3)*y[np.newaxis,:,np.newaxis]
    kfact = 4.*(kx[:,np.newaxis,np.newaxis]**3)*ky[np.newaxis,:,np.newaxis]
  elif (iterm == 5):
    rfact = (x[:,np.newaxis,np.newaxis]**3)*z[np.newaxis,np.newaxis,:]
    kfact = 4.*(kx[:,np.newaxis,np.newaxis]**3)*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 6):
    rfact = (y[np.newaxis,:,np.newaxis]**3)*x[:,np.newaxis,np.newaxis]
    kfact = 4.*(ky[np.newaxis,:,np.newaxis]**3)*kx[:,np.newaxis,np.newaxis]
  elif (iterm == 7):
    rfact = (y[np.newaxis,:,np.newaxis]**3)*z[np.newaxis,np.newaxis,:]
    kfact = (ky[np.newaxis,:,np.newaxis]**3)*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 8):
    rfact = (z[np.newaxis,np.newaxis,:]**3)*x[:,np.newaxis,np.newaxis]
    kfact = (kz[np.newaxis,np.newaxis,:]**3)*kx[:,np.newaxis,np.newaxis]
  elif (iterm == 9):
    rfact = (z[np.newaxis,np.newaxis,:]**3)*y[np.newaxis,:,np.newaxis]
    kfact = (kz[np.newaxis,np.newaxis,:]**3)*ky[np.newaxis,:,np.newaxis]
  elif (iterm == 10):
    rfact = (x[:,np.newaxis,np.newaxis]**2)*(y[np.newaxis,:,np.newaxis]**2)
    kfact = 6.*(kx[:,np.newaxis,np.newaxis]**2)*(ky[np.newaxis,:,np.newaxis]**2)
  elif (iterm == 11):
    rfact = (x[:,np.newaxis,np.newaxis]**2)*(z[np.newaxis,np.newaxis,:]**2)
    kfact = 6.*(kx[:,np.newaxis,np.newaxis]**2)*(kz[np.newaxis,np.newaxis,:]**2)
  elif (iterm == 12):
    rfact = (y[np.newaxis,:,np.newaxis]**2)*(z[np.newaxis,np.newaxis,:]**2)
    kfact = 6.*(ky[np.newaxis,:,np.newaxis]**2)*(kz[np.newaxis,np.newaxis,:]**2)
  elif (iterm == 13):
    rfact = (x[:,np.newaxis,np.newaxis]**2)*y[np.newaxis,:,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 12.*(kx[:,np.newaxis,np.newaxis]**2)*ky[np.newaxis,:,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 14):
    rfact = (y[np.newaxis,:,np.newaxis]**2)*x[:,np.newaxis,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 12.*(ky[np.newaxis,:,np.newaxis]**2)*kx[:,np.newaxis,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 15):
    rfact = (z[np.newaxis,np.newaxis,:]**2)*x[:,np.newaxis,np.newaxis]*y[np.newaxis,:,np.newaxis]
    kfact = 12.*(kz[np.newaxis,np.newaxis,:]**2)*kx[:,np.newaxis,np.newaxis]*ky[np.newaxis,:,np.newaxis]
  return rfact,kfact

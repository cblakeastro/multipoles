########################################################################
# Code to determine the window function multipoles W_l^2(s) for a      #
# curved sky, with a gridded survey window function and weight         #
# function.                                                            #
#                                                                      #
# The algorithm is based on Blake et al., 2018, MNRAS, submitted.      #
#                                                                      #
# Arguments:                                                           #
# smin,smax -- minimum,maximum separation [Mpc/h]                      #
# nsbin -- number of s bins                                            #
# nx,ny,nz -- size of gridded cuboid                                   #
# lx,ly,lz -- dimensions of gridded cuboid [Mpc/h]                     #
# x0,y0,z0 -- co-ordinate origin in cuboid co-ordinates [Mpc/h]        #
# weigrid -- gridded weight function                                   #
# wingrid -- gridded window function                                   #
#                                                                      #
# Returns:                                                             #
# sbin -- separations for array                                        #
# w0sq,w2sq,w4sq,w6sq,w8sq -- window function multipoles               #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
from scipy.special import jn
from scipy.special import sph_harm

def getpolewin(smin,smax,nsbin,nx,ny,nz,lx,ly,lz,x0,y0,z0,weigrid,wingrid):
# Initializations
  lmaxwin = 4 # Maximum window function multipole to compute
  print '\nComputing window function multipoles using curved sky...'
  print 'lmaxwin =',lmaxwin
  print 'smin =',smin,'smax =',smax,'ns =',nsbin
  nl = lmaxwin/2 + 1
  sbin = np.linspace(smin,smax,nsbin)
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
  sumw = np.sum(wingrid)
  sumwsq = nc*np.sum(((wingrid/sumw)*weigrid)**2)
# Pre-compute Bessel functions
  umin,umax = 0.1,np.amax(kgrid)*smax
  umod,jlmod = getjlmod(lmaxwin,umin,umax)
# Obtain spherical polar angles over the grid
  xthetagrid = np.arctan2(z[np.newaxis,np.newaxis,:],y[np.newaxis,:,np.newaxis])
  xphigrid = np.where(rgrid>0.,np.arccos(x[:,np.newaxis,np.newaxis]/rgrid),0.)
  kthetagrid = np.arctan2(kz[np.newaxis,np.newaxis,:],ky[np.newaxis,:,np.newaxis])
  kphigrid = np.where(kgrid>0.,np.arccos(kx[:,np.newaxis,np.newaxis]/kgrid),0.)
# FFT window function
  winspec = np.fft.fftn(weigrid*(wingrid/sumw))
  w0sq,w2sq,w4sq,w6sq,w8sq = np.zeros(nsbin),np.zeros(nsbin),np.zeros(nsbin),np.zeros(nsbin),np.zeros(nsbin)
  for il in range(nl):
    l = 2*il
# Compute window function as sum over m
    for m in range(-l,l+1):
      print 'Computing window function multipoles for l =',l,'m =',m,'...'
      xylmgrid = sph_harm(m,l,xthetagrid,xphigrid)
      slmgrid = np.fft.fftn(weigrid*(wingrid/sumw)*np.conj(xylmgrid))
      kylmgrid = sph_harm(m,l,kthetagrid,kphigrid)
      for i in range(nsbin):
        s = sbin[i]
        jlgrid = np.interp(kgrid*s,umod,jlmod[l,:])
        ctemp = np.sum(np.conj(winspec)*jlgrid*kylmgrid*slmgrid)
        temp = (4.*np.pi/sumwsq)*np.real(ctemp)
        if (il == 0):
          w0sq[i] += temp
        elif (il == 1):
          w2sq[i] -= temp # negative because of i^l factor
        elif (il == 2):
          w4sq[i] += temp
        elif (il == 3):
          w6sq[i] -= temp # negative because of i^l factor
        elif (il == 4):
          w8sq[i] += temp
  print '\nWindow function multipoles:'
  for i in range(nsbin):
    print sbin[i],'{:7.4f}'.format(w0sq[i]),'{:7.4f}'.format(w2sq[i]),'{:7.4f}'.format(w4sq[i]),'{:7.4f}'.format(w6sq[i]),'{:7.4f}'.format(w8sq[i])
  return sbin,w0sq,w2sq,w4sq,w6sq,w8sq

# Pre-compute Bessel functions
def getjlmod(lmax,umin,umax):
  du = 0.1
  umax1 = umax + lmax*np.pi
  nu = int(np.floor(umax1/du)) + 1
  print 'Pre-computing Bessel functions to l =',lmax,'...'
  print 'Using',nu,'points over',umin,'< u <',umax1,'...'
  umod = np.linspace(umin,umax1,nu)
  jlmod = np.empty((lmax+1,nu))
  for iu in range(nu):
    u = umod[iu]
    for l in range(lmax+1):
      jlmod[l,iu] = np.sqrt(np.pi/(2.*u))*jn(l+0.5,u)
  return umod,jlmod

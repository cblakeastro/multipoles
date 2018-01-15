########################################################################
# Code to estimate the covariance matrix of power spectrum multipoles  #
# in the Gaussian approximation.                                       #
#                                                                      #
# The algorithm is based on Blake et al., 2018, MNRAS, submitted.      #
#                                                                      #
# Arguments:                                                           #
# nx,ny,nz -- size of gridded cuboid                                   #
# lx,ly,lz -- dimensions of gridded cuboid [Mpc/h]                     #
# x0,y0,z0 -- co-ordinate origin in cuboid co-ordinates [Mpc/h]        #
# ngal -- number of galaxies in sample                                 #
# kmin,kmax -- minimum,maximum wavenumber for binning [h/Mpc]          #
# nkbin -- number of k bins                                            #
# nmax -- number of modes to use in each shell for covariance estimate #
# pk0,pk2,pk4 -- fiducial power spectrum multipoles [(Mpc/h)^3]        #
# weigrid -- gridded weight function                                   #
# wingrid -- gridded window function                                   #
#                                                                      #
# Returns:                                                             #
# pkcov -- covariance matrix of size (3*nkbin,3*nkbin), corresponding  #
#          to [P0(k1), P0(k2), ..., P2(k1), ..., P4(k1), ...]          #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
from scipy.special import sph_harm
import tools

def getpolecov(nx,ny,nz,lx,ly,lz,x0,y0,z0,ngal,kmin,kmax,nkbin,nmax,pk0,pk2,pk4,weigrid,wingrid):
  print '\nDetermining covariance matrix for power spectrum multipoles...'
  nl = 3    # Number of multipoles to compute in covariance
  nlmod = 2 # Number of multipoles in model
# Initializations
  doindep,dohalf = False,False
  kgrid,mugrid,indep = tools.getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf)
  ix,iy,iz = np.arange(nx),np.arange(ny),np.arange(nz)
  ixgrid = np.transpose(np.tile(ix,(nz,ny,1)),(2,1,0))
  iygrid = np.transpose(np.tile(iy,(nx,nz,1)),(0,2,1))
  izgrid = np.transpose(np.tile(iz,(ny,nx,1)),(1,0,2))
  vol,nc = lx*ly*lz,float(nx*ny*nz)
  vc = vol/nc
  winnorm = (ngal*nc)/(vol*np.sum(wingrid))
  weinorm = np.sqrt(nc/(vol*np.sum((weigrid*wingrid)**2)))/winnorm
  pkl = np.empty((3,nkbin))
  pkl[0,:],pkl[1,:],pkl[2,:] = pk0,pk2,pk4
# List of (kx,ky,kz) modes belonging to each bin
  bincount,ixlist,iylist,izlist = doixiyizlist(kgrid,ixgrid,iygrid,izgrid,kmin,kmax,nkbin,nmax)
# Obtain spherical polar angles over the grid
  dx,dy,dz = lx/nx,ly/ny,lz/nz
  x = dx*np.arange(nx) - x0
  y = dy*np.arange(ny) - y0
  z = dz*np.arange(nz) - z0
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  kgrid = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  xthetagrid = np.arctan2(z[np.newaxis,np.newaxis,:],y[np.newaxis,:,np.newaxis])
  xphigrid = np.where(rgrid>0.,np.arccos(x[:,np.newaxis,np.newaxis]/rgrid),0.)
  kthetagrid = np.arctan2(kz[np.newaxis,np.newaxis,:],ky[np.newaxis,:,np.newaxis])
  kphigrid = np.where(kgrid>0.,np.arccos(kx[:,np.newaxis,np.newaxis]/kgrid),0.)
# Generate a_l,l',l'' coefficients
  lppmax = 2*(nl+nlmod-2)
  nlpp = nl+nlmod-1
  g,alll = np.empty(lppmax+1),np.zeros((nl,nlmod,nlpp))
  g[0] = 1.
  for p in range(1,lppmax+1):
    g[p] = g[p-1]*(float(2*p-1)/float(p))
  for il in range(nl):
    l = 2*il
    for ilp in range(min(il+1,nlmod)):
      lp = 2*ilp
      for p in range(min(l,lp)+1):
        lpp = l+lp-2*p
        ilpp = lpp/2
        alll[il,ilp,ilpp] = (g[l-p]*g[p]*g[lp-p]/g[l+lp-p])*(float(2*l+2*lp-4*p+1)/float(2*l+2*lp-2*p+1))
# Pre-compute harmonic averages using selected modes
# Maximum value of lpp is l + lp, so need to generate to 2*lmax
  qqave1 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
  qsave1 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
  sqave1 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
  ssave1 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
  qqave2 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
  qsave2 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
  sqave2 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
  ssave2 = np.zeros((nlpp,2*lppmax+1,nlpp,2*lppmax+1,nkbin,nkbin))
# (0,0) terms
  q00spec = vc*np.fft.fftn((weinorm*weigrid*winnorm*wingrid)**2)
  s00spec = vc*np.fft.fftn(((weinorm*weigrid)**2)*winnorm*wingrid)
  for il1 in range(nlpp):
    l1 = 2*il1
    for im1 in range(2*l1+1):
      m1 = im1 - l1
      print 'Computing covariance for l =',l1,'m =',m1,'...'
      xylmspec1 = sph_harm(m1,l1,xthetagrid,xphigrid)
      qlmspec1 = vc*np.fft.fftn(((weinorm*weigrid*winnorm*wingrid)**2)*xylmspec1)
      slmspec1 = vc*np.fft.fftn(((weinorm*weigrid)**2)*winnorm*wingrid*xylmspec1)
      kylmspec1 = sph_harm(m1,l1,kthetagrid,kphigrid)
      for il2 in range(nlpp):
        l2 = 2*il2
        for im2 in range(2*l2+1):
          m2 = im2 - l2
          xylmspec2 = sph_harm(m2,l2,xthetagrid,xphigrid)
          qlmspec2 = vc*np.fft.fftn(((weinorm*weigrid*winnorm*wingrid)**2)*xylmspec2)
          slmspec2 = vc*np.fft.fftn(((weinorm*weigrid)**2)*winnorm*wingrid*xylmspec2)
          kylmspec2 = sph_harm(m2,l2,kthetagrid,kphigrid)
          qlmspec12 = vc*np.fft.fftn(((weinorm*weigrid*winnorm*wingrid)**2)*xylmspec1*np.conj(xylmspec2))
          slmspec12 = vc*np.fft.fftn(((weinorm*weigrid)**2)*winnorm*wingrid*xylmspec1*np.conj(xylmspec2))
# Obtain covariance between each pair of bins by mode-averaging
          for ibin in range(nkbin):
            for jbin in range(ibin,nkbin):
              if ((bincount[ibin] > 0) & (bincount[jbin] > 0)):
                qqave1[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,qlmspec1,qlmspec2,ixlist,iylist,izlist)
                qqave2[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,qlmspec12,q00spec,ixlist,iylist,izlist)
                qsave1[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,qlmspec1,slmspec2,ixlist,iylist,izlist)
                sqave1[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,slmspec1,qlmspec2,ixlist,iylist,izlist)
                qsave2[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,qlmspec12,s00spec,ixlist,iylist,izlist)
                sqave2[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,slmspec12,q00spec,ixlist,iylist,izlist)
                ssave1[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,slmspec1,slmspec2,ixlist,iylist,izlist)
                ssave2[il1,im1,il2,im2,ibin,jbin] = dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,slmspec12,s00spec,ixlist,iylist,izlist)
# Generate rest of bins
  for il1 in range(nlpp):
    l1 = 2*il1
    for im1 in range(2*l1+1):
      m1 = im1 - l1
      im1m = l1 - m1
      for il2 in range(nlpp):
        l2 = 2*il2
        for im2 in range(2*l2+1):
          m2 = im2 - l2
          im2m = l2 - m2
          for ibin in range(nkbin):
            for jbin in range(ibin):
              qqave1[il1,im1,il2,im2,ibin,jbin] = qqave1[il2,im2m,il1,im1m,jbin,ibin]
              qsave1[il1,im1,il2,im2,ibin,jbin] = sqave1[il2,im2m,il1,im1m,jbin,ibin]
              sqave1[il1,im1,il2,im2,ibin,jbin] = qsave1[il2,im2m,il1,im1m,jbin,ibin]
              ssave1[il1,im1,il2,im2,ibin,jbin] = ssave1[il2,im2m,il1,im1m,jbin,ibin]
              qqave2[il1,im1,il2,im2,ibin,jbin] = qqave2[il2,im2,il1,im1,jbin,ibin]
              qsave2[il1,im1,il2,im2,ibin,jbin] = qsave2[il2,im2,il1,im1,jbin,ibin]
              sqave2[il1,im1,il2,im2,ibin,jbin] = sqave2[il2,im2,il1,im1,jbin,ibin]
              ssave2[il1,im1,il2,im2,ibin,jbin] = ssave2[il2,im2,il1,im1,jbin,ibin]
# Convert averages into covariance
  pkcov = np.full((3*nkbin,3*nkbin),-1.)
  for ibin in range(nkbin):
    for jbin in range(nkbin):
      if ((bincount[ibin] > 0) & (bincount[jbin] > 0)):
        for il1 in range(nl):
          l1 = 2*il1
          for il2 in range(nl):
            l2 = 2*il2
# Sample variance
            qqsum1,qqsum2 = 0.,0.
            for il1p in range(nlmod):
              for il1pp in range(il1+il1p+1):
                l1pp = 2*il1pp
                for il2p in range(nlmod):
                  for il2pp in range(il2+il2p+1):
                    l2pp = 2*il2pp
                    qnorm1 = alll[il1,il1p,il1pp]*((4.*np.pi)/float(2*l1pp+1))
                    qnorm2 = alll[il2,il2p,il2pp]*((4.*np.pi)/float(2*l2pp+1))
                    for im1pp in range(2*l1pp+1):
                      for im2pp in range(2*l2pp+1):
                        qqsum1 += qnorm1*qnorm2*pkl[il1p,ibin]*pkl[il2p,jbin]*qqave1[il1pp,im1pp,il2pp,im2pp,ibin,jbin]
              for il1pp in range(il1+il1p+1):
                l1pp = 2*il1pp
                qnorm1 = alll[il1,il1p,il1pp]*((4.*np.pi)/float(2*l1pp+1))
                qnorm2 = (4.*np.pi)/float(2*l2+1)
                for im1pp in range(2*l1pp+1):
                  for im2 in range(2*l2+1):
                    qqsum2 += qnorm1*qnorm2*0.5*pkl[il1p,ibin]*np.sqrt(pkl[0,ibin]*pkl[0,jbin])*qqave2[il1pp,im1pp,il2,im2,ibin,jbin]
              for il1pp in range(il2+il1p+1):
                l1pp = 2*il1pp
                qnorm1 = alll[il2,il1p,il1pp]*((4.*np.pi)/float(2*l1pp+1))
                qnorm2 = (4.*np.pi)/float(2*l1+1)
                for im1pp in range(2*l1pp+1):
                  for im1 in range(2*l1+1):
                    qqsum2 += qnorm1*qnorm2*0.5*pkl[il1p,jbin]*np.sqrt(pkl[0,ibin]*pkl[0,jbin])*qqave2[il1pp,im1pp,il1,im1,jbin,ibin]
# Cross terms
            qssum1,qssum2 = 0.,0.
            for il1p in range(nlmod):
              for il1pp in range(il1+il1p+1):
                l1pp = 2*il1pp
                qnorm1 = alll[il1,il1p,il1pp]*((4.*np.pi)/float(2*l1pp+1))
                snorm2 = (4.*np.pi)/float(2*l2+1)
                for im1pp in range(2*l1pp+1):
                  for im2 in range(2*l2+1):
                    qssum1 += qnorm1*snorm2*pkl[il1p,ibin]*qsave1[il1pp,im1pp,il2,im2,ibin,jbin]
              for il1pp in range(il1+il1p+1):
                l1pp = 2*il1pp
                qnorm1 = alll[il1,il1p,il1pp]*((4.*np.pi)/float(2*l1pp+1))
                snorm2 = (4.*np.pi)/float(2*l2+1)
                for im1pp in range(2*l1pp+1):
                  for im2 in range(2*l2+1):
                    qssum2 += qnorm1*snorm2*0.5*pkl[il1p,ibin]*qsave2[il1pp,im1pp,il2,im2,ibin,jbin]
              for il1pp in range(il2+il1p+1):
                l1pp = 2*il1pp
                qnorm1 = alll[il2,il1p,il1pp]*((4.*np.pi)/float(2*l1pp+1))
                snorm2 = (4.*np.pi)/float(2*l1+1)
                for im1pp in range(2*l1pp+1):
                  for im1 in range(2*l1+1):
                    qssum2 += qnorm1*snorm2*0.5*pkl[il1p,jbin]*qsave2[il1pp,im1pp,il1,im1,jbin,ibin]
            sqsum1,sqsum2 = 0.,0.
            for im1 in range(2*l1+1):
              for il2p in range(nlmod):
                for il2pp in range(il2+il2p+1):
                  l2pp = 2*il2pp
                  snorm1 = (4.*np.pi)/float(2*l1+1)
                  qnorm2 = alll[il2,il2p,il2pp]*((4.*np.pi)/float(2*l2pp+1))
                  for im2pp in range(2*l2pp+1):
                    sqsum1 += snorm1*qnorm2*pkl[il2p,jbin]*sqave1[il1,im1,il2pp,im2pp,ibin,jbin]
              snorm1 = (4.*np.pi)/float(2*l1+1)
              qnorm2 = (4.*np.pi)/float(2*l2+1)
              for im2 in range(2*l2+1):
                temp = snorm1*qnorm2*np.sqrt(pkl[0,ibin]*pkl[0,jbin])*sqave2[il1,im1,il2,im2,ibin,jbin]
                sqsum2 += temp
# Shot noise
            sssum1,sssum2 = 0.,0.
            snorm1 = (4.*np.pi)/float(2*l1+1)
            snorm2 = (4.*np.pi)/float(2*l2+1)
            for im1 in range(2*l1+1):
              for im2 in range(2*l2+1):
                sssum1 += snorm1*snorm2*ssave1[il1,im1,il2,im2,ibin,jbin]
                sssum2 += snorm1*snorm2*ssave2[il1,im1,il2,im2,ibin,jbin]
            cov = float((2*l1+1)*(2*l2+1))*(qqsum1+qqsum2+qssum1+qssum2+sqsum1+sqsum2+sssum1+sssum2)
            pkcov[nkbin*il1+ibin,nkbin*il2+jbin] = cov
# Test that covariance is symmetric
  sym = True
  for i in range(3*nkbin):
    for j in range(i,3*nkbin):
      if (abs(pkcov[i,j]-pkcov[j,i]) > 0.1):
        print i+1,j+1,pkcov[i,j],pkcov[j,i]
        sym = False
  if (not sym):
    print '\n*** Warning: covariance not symmetric!!'
# Display measurements:
  dk = (kmax-kmin)/nkbin
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  print '\nDiagonal errors from covariance:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:9.1f}'.format(np.sqrt(pkcov[ik,ik])),'{:9.1f}'.format(np.sqrt(pkcov[nkbin+ik,nkbin+ik])),'{:9.1f}'.format(np.sqrt(pkcov[2*nkbin+ik,2*nkbin+ik]))
  return pkcov

# Randomly select modes within each spherical shell
def doixiyizlist(kgrid,ixgrid,iygrid,izgrid,kmin,kmax,nkbin,nmax):
  dk = (kmax-kmin)/nkbin
  k1,k2 = np.linspace(kmin,kmax-dk,nkbin),np.linspace(kmin+dk,kmax,nkbin)
  nbin = nkbin
  bincount = np.empty(nbin,dtype=int)
  ixlist,iylist,izlist = np.empty((nbin,nmax),dtype=int),np.empty((nbin,nmax),dtype=int),np.empty((nbin,nmax),dtype=int)
  for ik in range(nkbin):
    cut = (kgrid > 0.) & (kgrid >= k1[ik]) & (kgrid < k2[ik])
    bincount[ik] = np.count_nonzero(cut)
    print ik+1,bincount[ik]
    if (bincount[ik] > nmax):
      ind = np.random.choice(bincount[ik],nmax,replace=False)
      ixlist[ik,:] = ixgrid[cut][ind]
      iylist[ik,:] = iygrid[cut][ind]
      izlist[ik,:] = izgrid[cut][ind]
    else:
      ixlist[ik,:bincount[ik]] = ixgrid[cut]
      iylist[ik,:bincount[ik]] = iygrid[cut]
      izlist[ik,:bincount[ik]] = izgrid[cut]
  return bincount,ixlist,iylist,izlist

# Take average of Y1(k)^*.f(k-k').Y2(k').g(k-k')^*
def dopolecovharmsum(ibin,jbin,bincount,nmax,nx,ny,nz,kylmspec1,kylmspec2,flmspec,glmspec,ixlist,iylist,izlist):
  imax,jmax = min(bincount[ibin],nmax),min(bincount[jbin],nmax)
  ixlisti,ixlistj = np.meshgrid(ixlist[ibin,:imax],ixlist[jbin,:jmax],indexing='ij')
  iylisti,iylistj = np.meshgrid(iylist[ibin,:imax],iylist[jbin,:jmax],indexing='ij')
  izlisti,izlistj = np.meshgrid(izlist[ibin,:imax],izlist[jbin,:jmax],indexing='ij')
  mx,my,mz = ixlisti-ixlistj,iylisti-iylistj,izlisti-izlistj
  mx[mx < 0] += nx
  my[my < 0] += ny
  mz[mz < 0] += nz
  cut = ((mx != 0) | (my != 0) | (mz != 0))
  pkcov = np.sum(np.real(np.conj(kylmspec1[[ixlisti[cut],iylisti[cut],izlisti[cut]]])*kylmspec2[[ixlistj[cut],iylistj[cut],izlistj[cut]]]*flmspec[[mx[cut],my[cut],mz[cut]]]*np.conj(glmspec[[mx[cut],my[cut],mz[cut]]])))
  pkcov *= 1./float(imax*jmax)
  if (jbin == ibin):
    ylm1ylm2ave = np.mean(np.conj(kylmspec1[[ixlisti,iylisti,izlisti]])*kylmspec2[[ixlisti,iylisti,izlisti]])
    pkcov += np.real(ylm1ylm2ave*flmspec[0,0,0]*np.conj(glmspec[0,0,0]))/bincount[ibin]
  return pkcov

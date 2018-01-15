import numpy as np

# Read in selection function
def readwin(winfile):
  print '\nReading in window function...'
  print winfile
  f = open(winfile,'r')
  f.readline()
  fields = f.readline().split()
  nx,ny,nz = int(fields[0]),int(fields[1]),int(fields[2])
  if (len(fields) < 9):
    f.readline()
  wingrid = np.zeros(shape=(nx,ny,nz))
  for iz in range(nz):
    for iy in range(ny):
      for ix in range(nx):
        wingrid[ix,iy,iz] = float(f.readline())
  f.close()
  print 'Number of randoms =','{:.2e}'.format(np.sum(wingrid))
  return wingrid,nx,ny,nz

# Grid galaxy distribution
def discret(xpos,ypos,zpos,nx,ny,nz,lx,ly,lz,x0,y0,z0):
  print '\nGridding',len(xpos),'objects...'
  datgrid,edges = np.histogramdd(np.vstack([xpos+x0,ypos+y0,zpos+z0]).transpose(),bins=(nx,ny,nz),range=((0.,lx),(0.,ly),(0.,lz)))
  return datgrid

# Bin 3D power spectrum in angle-averaged bins
def binpk(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf):
  print 'Binning in angle-averaged bins...'
  kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf)
  pkspec = pkspec[indep == True]
  kspec = kspec[indep == True]
  ikbin = np.digitize(kspec,np.linspace(kmin,kmax,nkbin+1))
  nmodes,pk = np.zeros(nkbin,dtype=int),np.full(nkbin,-1.)
  for ik in range(nkbin):
    nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
    if (nmodes[ik] > 0):
      pk[ik] = np.mean(pkspec[ikbin == ik+1])
  return pk,nmodes

# Obtain 3D grid of k-modes
def getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf):
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
  if (dohalf):
    kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:nz/2+1]
    indep = np.full((nx,ny,nz/2+1),True,dtype=bool)
    if (doindep):
      indep = getindep(nx,ny,nz)
  else:
    kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)
    indep = np.full((nx,ny,nz),True,dtype=bool)
  indep[0,0,0] = False
  kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  kspec[0,0,0] = 1.
  muspec = np.absolute(kx[:,np.newaxis,np.newaxis])/kspec
  kspec[0,0,0] = 0.
  return kspec,muspec,indep

# Array of independent 3D modes
def getindep(nx,ny,nz):
  indep = np.full((nx,ny,nz/2+1),False,dtype=bool)
  indep[:,:,1:nz/2] = True
  indep[1:nx/2,:,0] = True
  indep[1:nx/2,:,nz/2] = True
  indep[0,1:ny/2,0] = True
  indep[0,1:ny/2,nz/2] = True
  indep[nx/2,1:ny/2,0] = True
  indep[nx/2,1:ny/2,nz/2] = True
  indep[nx/2,0,0] = True
  indep[0,ny/2,0] = True
  indep[nx/2,ny/2,0] = True
  indep[0,0,nz/2] = True
  indep[nx/2,0,nz/2] = True
  indep[0,ny/2,nz/2] = True
  indep[nx/2,ny/2,nz/2] = True
  return indep

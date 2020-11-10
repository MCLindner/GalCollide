#!/usr/bin/env python

"""ObservationFrame.py: Description."""

__author__ = "Michael Lindner"

import pynbody
import numpy as np
import ObservationParameters as p

L = p.L
V = p.V
snapshot = p.snapshot

# Rotate from simulation coordinates to observation coordinates
# R=LMr+Rc,V=VMv+Vc.

# void rotmatrix(matrix rmat, real xrot, real yrot, real zrot)
# {
#   real sx = rsinD(xrot), sy = rsinD(yrot), sz = rsinD(zrot);
#   real cx = rcosD(xrot), cy = rcosD(yrot), cz = rcosD(zrot);
#   matrix xmat, ymat, zmat, tmp1;

#   xmat[0][0] = 1.0;    xmat[0][1] = 0.0;    xmat[0][2] = 0.0;
#   xmat[1][0] = 0.0;    xmat[1][1] =  cx;    xmat[1][2] =  sx;
#   xmat[2][0] = 0.0;    xmat[2][1] = -sx;    xmat[2][2] =  cx;

#   ymat[0][0] =  cy;    ymat[0][1] = 0.0;    ymat[0][2] = -sy;
#   ymat[1][0] = 0.0;    ymat[1][1] = 1.0;    ymat[1][2] = 0.0;
#   ymat[2][0] =  sy;    ymat[2][1] = 0.0;    ymat[2][2] =  cy;

#   zmat[0][0] =  cz;    zmat[0][1] =  sz;    zmat[0][2] = 0.0;
#   zmat[1][0] = -sz;    zmat[1][1] =  cz;    zmat[1][2] = 0.0;
#   zmat[2][0] = 0.0;    zmat[2][1] = 0.0;    zmat[2][2] = 1.0;
#   MULM(tmp1, xmat, ymat);
#   MULM(rmat, zmat, tmp1);
# }

xrot = p.thetax
yrot = p.thetay
zrot = p.thetaz

sx = np.sin(xrot)
sy = np.sin(yrot)
sz = np.sin(zrot)

cx = np.cos(xrot)
cy = np.cos(yrot)
cz = np.cos(zrot)

# amat = np.matrix([[c_a, s_a, 0.0],
#                   [-s_a, c_a, 0.0],
#                   [0.0, 0.0, 1.0]])

# bmat = np.matrix([[1.0, 0.0, 0.0],
#                   [0.0, c_b, s_b],
#                   [0.0, -s_b, c_b]])

# gmat = np.matrix([[c_g, s_g, 0.0],
#                   [-s_g, c_g, 0.0],
#                   [0.0, 0.0, 1.0]])

xmat = np.empty((3, 3))
ymat = np.empty((3, 3))
zmat = np.empty((3, 3))

xmat[0][0] = 1.0;    xmat[0][1] = 0.0;    xmat[0][2] = 0.0;
xmat[1][0] = 0.0;    xmat[1][1] =  cx;    xmat[1][2] =  sx;
xmat[2][0] = 0.0;    xmat[2][1] = -sx;    xmat[2][2] =  cx;

ymat[0][0] =  cy;    ymat[0][1] = 0.0;    ymat[0][2] = -sy;
ymat[1][0] = 0.0;    ymat[1][1] = 1.0;    ymat[1][2] = 0.0;
ymat[2][0] =  sy;    ymat[2][1] = 0.0;    ymat[2][2] =  cy;

zmat[0][0] =  cz;    zmat[0][1] =  sz;    zmat[0][2] = 0.0;
zmat[1][0] = -sz;    zmat[1][1] =  cz;    zmat[1][2] = 0.0;
zmat[2][0] = 0.0;    zmat[2][1] = 0.0;    zmat[2][2] = 1.0;

tmp1 = np.matmul(xmat, ymat)
rmat = np.matmul(zmat, tmp1)

MP = L * rmat
MV = V * rmat


def transformP(row):
    row = (MP * np.matrix(row).transpose())
    return(row)


def transformV(row):
    row = (MV * np.matrix(row).transpose())
    return(row)


lengths = {}
for fam in snapshot.families():
    lengths[fam.name] = len(snapshot[fam])

scaled_snapshot = pynbody.new(**lengths)

for fam in snapshot.families():
    print("Scaling family " + str(fam))
    snap_fam = snapshot[fam]

    snap_fam['pos'] = np.apply_along_axis(transformP, 1, snap_fam['pos'])
    snap_fam['vel'] = np.apply_along_axis(transformV, 1, snap_fam['vel'])

    scaled_snapshot[fam][:len(snap_fam)]['pos'] = snap_fam['pos'].in_units('kpc')
    scaled_snapshot[fam][:len(snap_fam)]['vel'] = snap_fam['vel'].in_units('.9778 km s**-1')
    scaled_snapshot[fam][:len(snap_fam)]['mass'] = snap_fam['mass'].in_units('2.2222858e5 Msol')
    scaled_snapshot[fam][:len(snap_fam)]['rho'] = snap_fam['rho'].in_units('2.2222858e5 Msol kpc**-3')
    scaled_snapshot[fam][:len(snap_fam)]['eps'] = snap_fam['eps'].in_units('kpc')

scaled_snapshot.s['tform'] = snapshot.s['tform']
scaled_snapshot.s['metals'] = snapshot.s['metals']
scaled_snapshot.properties['time'] = snapshot.properties['time']

scaled_snapshot.write(filename='scaled.tipsy', fmt=pynbody.tipsy.TipsySnap, cosmological=False)

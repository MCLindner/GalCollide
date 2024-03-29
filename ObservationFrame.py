#!/usr/bin/env python

"""ObservationFrame.py: Description."""

__author__ = "Michael Lindner"

import pynbody
import numpy as np
import ObservationParameters as p
import astropy.units as u

# snapshot = p.snapshot
dMsolUnit = 2.32503e9 * u.solMass
dKpcUnit = 1 * u.kpc
timeUnit = 9.7792e6 * u.yr
velUnit = 100 * u.km / u.s

# TODO: get rid of globals

xrot = p.thetax
yrot = p.thetay
zrot = p.thetaz

s_x = np.sin(xrot)
s_y = np.sin(yrot)
s_z = np.sin(zrot)

c_x = np.cos(xrot)
c_y = np.cos(yrot)
c_z = np.cos(zrot)

xmat = np.matrix([[1.0,  0.0,  0.0],
                  [0.0,  c_x,  s_x],
                  [0.0, -s_x,  c_x]])

ymat = np.matrix([[c_y,  0.0, -s_y],
                  [0.0,  1.0,  0.0],
                  [s_y,  0.0,  c_y]])

zmat = np.matrix([[c_z,  s_z,  0.0],
                  [-s_z, c_z,  0.0],
                  [0.0,  0.0,  1.0]])

tmp1 = np.matmul(xmat, ymat)
rmat = np.matmul(zmat, tmp1)


def transformP(row):
    row = (rmat * np.matrix(row).transpose())
    return(row)


def transformV(row):
    row = (rmat * np.matrix(row).transpose())
    return(row)


def obsTransform(snapshot):
    lengths = {}
    for fam in snapshot.families():
        lengths[fam.name] = len(snapshot[fam])

    scaled_snapshot = pynbody.new(**lengths)

    for fam in snapshot.families():
        print("Scaling family " + str(fam))
        snap_fam = snapshot[fam]

        snap_fam['pos'] = np.apply_along_axis(transformP, 1, snap_fam['pos'])
        snap_fam['vel'] = np.apply_along_axis(transformV, 1, snap_fam['vel'])

        scaled_snapshot[fam][:len(snap_fam)]['pos'] = snap_fam['pos'].in_units(str(dKpcUnit.value) + " kpc")
        scaled_snapshot[fam][:len(snap_fam)]['vel'] = snap_fam['vel'].in_units(str(velUnit.value) + " km s**-1")
        scaled_snapshot[fam][:len(snap_fam)]['mass'] = snap_fam['mass'].in_units(str(dMsolUnit.value) + " Msol")
        scaled_snapshot[fam][:len(snap_fam)]['rho'] = snap_fam['rho'].in_units(str((dMsolUnit / (dKpcUnit**3)).value) + " Msol kpc**-3")
        scaled_snapshot[fam][:len(snap_fam)]['eps'] = snap_fam['eps'].in_units('kpc')

    if str(fam) == "gas":
        # temp starts out uniform
        scaled_snapshot[fam][:len(snap_fam)]["temp"] = snap_fam["temp"][:]
        scaled_snapshot[fam][len(snap_fam):]["temp"] = snap_fam["temp"][:]
    else:
        pass

    return scaled_snapshot

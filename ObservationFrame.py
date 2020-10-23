#!/usr/bin/env python

"""ObservationFrame.py: Description."""

__author__ = "Michael Lindner"

import pynbody
import numpy as np
import ObervationParameters.py as p

L = p.L
V = p.V
snapshot = p.snapshot
L = 39.5
V = 165

# Rotate from simulation coordinates to observation coordinates
# R=LMr+Rc,V=VMv+Vc.

w1 = p.w
i1 = p.i
Omega1 = p.Omega

MP = L * np.matrix([[np.cos(Omega1)*np.cos(w1)-np.sin(Omega1)*np.cos(i1)*np.sin(w1), -np.cos(Omega1)*np.sin(w1)-np.sin(Omega1)*np.cos(i1)*np.cos(w1), np.sin(Omega1)*np.sin(i1)],
                   [np.sin(Omega1)*np.cos(w1)+np.cos(Omega1)*np.cos(i1)*np.sin(w1), -np.sin(Omega1)*np.sin(w1)+np.cos(Omega1)*np.cos(i1)*np.cos(w1), -np.cos(Omega1)*np.sin(i1)],
                   [np.sin(i1)*np.sin(w1), np.sin(i1)*np.cos(w1), np.cos(i1)]]) 

MV = V * np.matrix([[np.cos(Omega1)*np.cos(w1)-np.sin(Omega1)*np.cos(i1)*np.sin(w1), -np.cos(Omega1)*np.sin(w1)-np.sin(Omega1)*np.cos(i1)*np.cos(w1), np.sin(Omega1)*np.sin(i1)],
                   [np.sin(Omega1)*np.cos(w1)+np.cos(Omega1)*np.cos(i1)*np.sin(w1), -np.sin(Omega1)*np.sin(w1)+np.cos(Omega1)*np.cos(i1)*np.cos(w1), -np.cos(Omega1)*np.sin(i1)],
                   [np.sin(i1)*np.sin(w1), np.sin(i1)*np.cos(w1), np.cos(i1)]]) 


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

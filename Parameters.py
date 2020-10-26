#!/usr/bin/env python

import pynbody

# Paths to tipsy files for galaxy models
Gal1 = pynbody.load('./InitialConditions/initMW')
Gal2 = pynbody.load('./InitialConditions/initMW2')

# dDelta for changa param file
dDelta = .01

# Perigalactic distance in kpc
d_perigalactic = 14.8

# Initial separation distance in kpc
initial_separation = 10**2

# Eccentricity of system
eccentricity = 0.99

# Time in Myr since first perigalacticon passage of desired output snapshot
time = 175

# Output file name
writename = "TheMiceV2.tipsy"

# Euler angles to transform each galaxy by
Omega1, w1, i1 = 0, 4.10152, 0.261799
Omega2, w2, i2 = 0, 3.49066, 0.436332

# Transform galaxies by Euler angles
transform = True

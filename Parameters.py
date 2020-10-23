#!/usr/bin/env python

"""Main.py: Description."""

__author__ = "Michael Lindner"

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

# Time since first perigalacticon passage in Myr
time = 250

# Output file name
writename = "TheMiceV2.tipsy"

# Euler angles to transform each galaxy by
Omega1, w1, i1 = 0, 4.10152, 0.261799
Omega2, w2, i2 = 0, 3.49066, 0.436332

# Viewing angle relative to orbital plane
thetaX, thetaY, thetaZ = 78, -44, -130

# Transform galaxies by Euler angles
transform = True

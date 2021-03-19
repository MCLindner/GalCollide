#!/usr/bin/env python

import pynbody

# Paths to tipsy files for galaxy models
Gal1 = pynbody.load('./InitialConditions/GasHaloICV4')
Gal2 = pynbody.load('./InitialConditions/GasHaloICV4')

# dDelta for changa param file
dDelta = .01

# Perigalactic distance in kpc
d_perigalactic = 14.8

# Initial separation distance in kpc
initial_separation = 200

# Eccentricity of system
eccentricity = 0.99

# Time in Myr since first perigalacticon passage of desired output snapshot
time = 175

# Dynamical mass of the system in kg
mDyn = 1.312687200755e+42

# Output file name
writename = "MiceScaledMassV3"

# Euler angles to transform each galaxy by
W1, w1, i1 = 0, 4.10152, 0.261799
W2, w2, i2 = 0, 3.49066, 0.436332

# Transform galaxies by Euler angles
transform = True

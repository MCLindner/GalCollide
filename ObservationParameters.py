#!/usr/bin/env python

import pynbody

# Tipsy snapshot to transform into observation frame
snapshot = pynbody.load('./Data/TheMiceV3.tipsy.002601')

# Length scale factor
L = 39.5

# Velocity scale factor
V = 165

# Viewing angle relative to orbital plane
thetax = 1.36136
thetay = -0.767945
thetaz = -2.26893

thetay = -0

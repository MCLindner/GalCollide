#!/usr/bin/env python

import pynbody

# Tipsy snapshot to transform into observation frame
snapshot = pynbody.load('./Data/TheMice.tipsy.005000')

# Length scale factor
L = 39.5

# Velocity scale factor
V = 165

# Viewing angle relative to orbital plane
w = 1.36136
i = -0.767945
Omega = -2.26893

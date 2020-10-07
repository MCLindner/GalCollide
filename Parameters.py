import pynbody

# Paths to tipsy files for galaxy models
Gal1 = pynbody.load('./InitialConditions/initMW')
Gal2 = pynbody.load('./InitialConditions/initMW2')

# Perigalactic distance in kpc
d_perigalactic = 23

# Initial separation distance in kpc
initial_separation = 276

# Eccentricity of system
eccentricity = 0.6

# Output file name
writename = "testing.tipsy"

# Euler angles to transform each galaxy by
Omega1, w1, i1 = 0, 4.71239, 0.261799
Omega2, w2, i2 = 0, 4.71239, 1.0472

# Viewing angle relative to orbital plane
thetaX, thetaY, thetaZ = 0, 0, 0

# Transform galaxies by Euler angles
transform = True

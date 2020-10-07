import pynbody.plot.sph as sph
import matplotlib.pylab as plt
import astropy.units as u
import Parameters as p
from Combine import GalCombine

Gal1 = p.Gal1
Gal2 = p.Gal2
d_perigalactic = p.d_perigalactic
inital_separation = p.initial_separation
eccentricity = p.eccentricity
writename = p.writename
Omega1 = p.Omega1
w1 = p.w1
i1 = p.i1
Omega2 = p.Omega2
w2 = p.w2
i2 = p.i2
transform = p.transform

two_bodys = GalCombine(Gal1, Gal2,
                       d_perigalactic, inital_separation, eccentricity,
                       writename, Omega1, w1, i1, Omega2, w2, i2,
                       transform)

two_bodys.make_param_file()
two_bodys.make_director_file()
combined = two_bodys.combine()

sph.velocity_image(combined, vector_color="black", qty="rho",
                   width=5000, cmap="Blues", denoise=False,
                   approximate_fast=False, show_cbar=False)

# test1 = two_bodys.solve_ivp(Gal1)
# test2 = two_bodys.solve_ivp(Gal2)

# Mass1 = float(Gal1['mass'].sum().in_units('kg')) * u.kg
# Mass2 = float(Gal2['mass'].sum().in_units('kg')) * u.kg
# MassTot = Mass1 + Mass2


# x1 = (test1['y'][0] * u.m).to(u.kpc).value * Mass1 / MassTot
# x2 = (test2['y'][0] * u.m).to(u.kpc).value * Mass2 / MassTot
# y1 = (test1['y'][1] * u.m).to(u.kpc).value * Mass1 / MassTot
# y2 = (test2['y'][1] * u.m).to(u.kpc).value * Mass2 / MassTot

# fig, ax = plt.subplots(2)
# ax[0].plot(test1['t'], y1)
# ax[1].plot(test2['t'], y2)
# ax[0].set_ylabel('y')
# ax[0].set_xlabel('t')

# fig, ax = plt.subplots(figsize=(15, 15))

# ax.set_ylim(-50, 50)
# ax.set_xlim(-50, 50)

# ax.scatter(x1, y1, c='b')
# ax.scatter(x2, y2, c='r')
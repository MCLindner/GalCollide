#!/usr/bin/env python

"""Combine.py: Class for interaction between two galaxies."""

__author__ = "Michael Lindner"

import pynbody
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import astropy.units as u
import astropy.constants as const


class GalCombine:
    """
    Create an initial condition for two galaxies in a Keplerian orbit.

    Attributes
    ----------
    Gal1 : PyNBody TipsySnap
        Tipsy snapshot of first galaxy model.

    Gal2 : PyNbody TipsySnap
        Tipsy snapshot of second galaxy model.

    dDelta : float/int
        dDelta for changa param file.

    d_perigalactic : float/int
        Perigalactic distance given in kpc.

    initial_separation : float/int
        Initial separation distance between galaxies in kpc

    eccentricity : float/int
        Eccentricity of the system.

    time : float/int
        Time since (or until if negative) first perigalacticon passage in Myr.

    mDyn : float/int
        Desired dynamical mass of the system given in kg.

    writename : string
        Output file name.

    W1, w1, i1 : float/int
        Euler angles for galaxy 1.

    W2, w2, i2 : float/int
        Euler angles for galaxy 2.

    transform : Boolean
        Whether or not to transform each galaxy by Euler angles.

    Public methods
    ----------
    eccentric_anomalies() :

    calculate_ICs() :
        Calculates intial positions and velocies of galaxies to place them on a
        Keplerian orbit as specified by perigalactic distance, initial
        separation,eccentricity, the mass of the galaxies, and time since first
        perigalactivon passage.

    combine() :
        Generates the initial condition tipsy file by combining galaxy
        models with initial positions and velocites given by calculate_ICs().

    get_period() :
        Gets the period of the orbit of the system based upon orbital
        parameters.

    get_t_out() :
        First calculates how long it has been since first perigalacticon
        passage at the time of the initial snapshot, then calculates the
        output time for the simulation.

    make_param_file() :
        Creates the param file for use in ChaNGa.

    make_director_file() :
        Creates the director file for changa movie output. If gas is used
        changes director file appropriately.

    make_info_file() :

    solve_ivp() :
        Tests that the initial condition is
        correct by solving the two body problem.

    """

    def __init__(self, Gal1, Gal2, dDelta,
                 d_perigalactic, inital_separation, eccentricity, time,
                 mDyn, writename, W1, w1, i1, W2, w2, i2, transform):
        self.Gal1 = Gal1
        self.Gal2 = Gal2
        self.dDelta = dDelta
        self.d_perigalactic = d_perigalactic * u.kpc
        self.eccentricity = eccentricity
        self.inital_separation = inital_separation * u.kpc
        self.time = (time * u.Myr)
        self.writename = writename
        self.W1 = W1
        self.w1 = w1
        self.i1 = i1
        self.W2 = W2
        self.w2 = w2
        self.i2 = i2
        self.transform = transform

        self.G = const.G

        # simulation units for param file
        # TODO: find a way to grab from GalactICs
        self.dMsolUnit = 2.32503e9 * u.solMass
        self.dKpcUnit = 1 * u.kpc
        self.timeUnit = 9.7792e6 * u.yr
        self.velUnit = 100 * u.km / u.s

        self.semi_major_axis = (-d_perigalactic / (eccentricity - 1)) * u.kpc
        self.semi_minor_axis = (-d_perigalactic / (eccentricity - 1)
                                ) * np.sqrt(1 - eccentricity**2) * u.kpc
        # Bring in with units, could be a little redundant
        self.Mass1 = float(self.Gal1["mass"].sum().in_units("kg")) * u.kg
        self.Mass2 = float(self.Gal2["mass"].sum().in_units("kg")) * u.kg
        self.m_tot = self.Mass1 + self.Mass2

        self.use_gas = False

        self.mDyn = mDyn

        self.massScale = self.mDyn / self.m_tot.value

        self.Mass1_scaled = self.Mass1 * self.massScale
        self.Mass2_scaled = self.Mass2 * self.massScale
        self.m_tot_scaled = self.Mass1_scaled + self.Mass2_scaled

    def eccentric_anomalies(self):
        """
        Uses scipy.optimize.brenq to find the eccentric anomalies of
        each of the galaxies in the system given the input parameters.

        Returns
        ----------

        E1, E2 : float
        """
        def f(E, a, b, e, r):
            return ((np.sqrt((a * (np.cos(E) - e))**2 + (b * np.sin(E))**2) - r)).value

        try:
            E1 = -brentq(f, 0, np.pi, xtol=10e-6,
                         args=(
                             self.semi_major_axis,
                             self.semi_minor_axis,
                             self.eccentricity,
                             self.inital_separation)
                         )
        except ValueError:
            print("Error: semi-major axis may be larger than apoapsis")
            exit()

        # Find E2
        try:
            E2 = -brentq(f, 0, np.pi, xtol=10e-6,
                         args=(
                             self.semi_major_axis,
                             self.semi_minor_axis,
                             self.eccentricity,
                             self.inital_separation)
                         )
        except ValueError:
            print("Error: semi-major axis may be larger than apoapsis")
            exit()

        return E1, E2

    def calculate_ICs(self, which_gal):
        """
        Finds the velocity componants in cartesian that, given the
        parameters passed into the class, would result
        in a Keplerian two-body orbit.

        Parameters
        ----------

        Returns
        ----------
        x1, y1, vx1, vy1 : float

        or

        x1, y1, vx1, vy1 : float

        """
        # Constants
        k = np.sqrt(self.G * self.m_tot_scaled)

        # Derived parameters
        n = k * self.semi_major_axis**(-3 / 2)

        print("Finding eccentric anomalies")
        E1, E2 = self.eccentric_anomalies()
        print("Done")
        print(E1, E2)
        print("")

        # Gal1 ###############################################################

        # Find positions
        X1 = self.semi_major_axis * (np.cos(E1) - self.eccentricity)
        Y1 = self.semi_minor_axis * np.sin(E1)

        # Find velocity componants
        vX1 = (-self.semi_major_axis * n * np.sin(E1)) / \
            (1 - self.eccentricity * np.cos(E1))
        vY1 = (self.semi_minor_axis * n * np.cos(E1)) / \
            (1 - self.eccentricity * np.cos(E1))

        # Transition from position in orbital plane to position in space
        q1 = -1
        x1 = q1 * X1
        y1 = q1 * Y1
        vx1 = q1 * vX1
        vy1 = q1 * vY1

        # Gal2 ###############################################################

        # Find positions
        X2 = self.semi_major_axis * (np.cos(E2) - self.eccentricity)
        Y2 = self.semi_minor_axis * np.sin(E2)

        # Find velocity componants
        vX2 = (-self.semi_major_axis * n * np.sin(E2)) / \
            (1 - self.eccentricity * np.cos(E2))
        vY2 = (self.semi_minor_axis * n * np.cos(E2)) / \
            (1 - self.eccentricity * np.cos(E2))

        # Transition from position in orbital plane to position in space
        q2 = 1
        x2 = q2 * X2
        y2 = q2 * Y2
        vx2 = q2 * vX2
        vy2 = q2 * vY2

        # Check that the center of mass is as it should be
        self.xcom = (x1 * self.Mass1_scaled + x2 * self.Mass2_scaled) / self.m_tot_scaled
        self.ycom = (y1 * self.Mass1_scaled + y2 * self.Mass2_scaled) / self.m_tot_scaled
        print("xcom = " + str(self.xcom))
        print("ycom = " + str(self.ycom))

        if which_gal == self.Gal1:
            return x1, y1, vx1, vy1
        elif which_gal == self.Gal2:
            return x2, y2, vx2, vy2

    def get_period(self):
        """
        Returns the period of orbit in seconds based on Kepler"s laws.

        Returns
        ----------

        T : float
        """
        k = np.sqrt(self.G * self.m_tot_scaled)
        a = self.semi_major_axis
        T = np.sqrt(((4 * np.pi**2) / (k**2)) * a**3)
        print("T:" + str(T.decompose()))
        print("")

        return T

    def get_t_out(self):
        """
        Finds t_out, the ammount the time in seconds that the galaxies orbit
        until they reach time since first perigalacticon passage  given
        in Parameters.py. Used to determine NSteps in make_param_file()

        Returns
        ----------
        """
        E1, E2 = self.eccentric_anomalies()
        M1 = E1 - self.eccentricity * np.sin(E1)
        # t - tau = time since first pericenter passage = t_now
        # scaled mass here
        t_now = M1 / np.sqrt((self.G * self.Mass1_scaled) / self.semi_major_axis**3)
        print("tnow:" + str(t_now.to(u.Gyr)))
        print("")
        t_since_passage_obs = (self.time).to(u.s)
        t_out = + t_since_passage_obs - (t_now).to(u.s)

        return t_out

    def _initial_conds(self, which_gal):
        """
        Returns the initial positions and velocities for galaxy given
        by which_gal. Utility function.

        Parameters
        ----------

        Returns
        ----------
        """
        x, y, vx, vy = self.calculate_ICs(which_gal)
        print(str(which_gal) + ": x,y,vx,vy = " + str((x, y, vx.decompose(),
                                                       vy.decompose())))
        print("")

        return x, y, vx, vy

    def _equations(self, t, p):
        """
        Splits second order diff eqquations of motion into four
        first order differential equations.

        Parameters
        ----------

        Returns
        ----------
        """
        r = np.sqrt((p[0])**2 + (p[1])**2)

        # Split the diff EQs
        #         u1 = x
        #         u2 = y
        #         u3 = dx
        #         u4 = dy

        du1 = p[2]
        du2 = p[3]
        du3 = -(((self.G * self.m_tot_scaled).value) / r**3) * (p[0])
        du4 = -(((self.G * self.m_tot_scaled).value) / r**3) * (p[1])

        return [du1, du2, du3, du4]

    def solve_ivp(self, which_gal):
        """
        Test that the initial condition is correct by solving
        the two-body problem for the system.

        Parameters
        ----------

        Returns
        ----------
        """
        nsteps = 10000
        period = self.get_period().to(u.s).value
        t_eval = np.linspace(0, period, nsteps)

        if which_gal == self.Gal1:
            x1, y1, vx1, vy1 = self._initial_conds(self.Gal1)
            sol = solve_ivp(self._equations, [0, period],
                            [x1.to(u.m).value, y1.to(u.m).value,
                             vx1.to(u.m / u.s).value, vy1.to(u.m / u.s).value],
                            t_eval=t_eval, rtol=1e-6)
        elif which_gal == self.Gal2:
            x2, y2, vx2, vy2 = self._initial_conds(self.Gal2)
            sol = solve_ivp(self._equations, [0, period],
                            [x2.to(u.m).value, y2.to(u.m).value,
                             vx2.to(u.m / u.s).value, vy2.to(u.m / u.s).value],
                            t_eval=t_eval, rtol=1e-6)

        return sol

    def _rmat(self, a, b, g):
        """
        Rotation matrix used for transforming galaxies.

        Parameters
        ----------

        Returns
        ----------
        """
        c_a = np.cos(a)
        c_b = np.cos(b)
        c_g = np.cos(g)

        s_a = np.sin(a)
        s_b = np.sin(b)
        s_g = np.sin(g)

        amat = np.matrix([[c_a,  s_a,  0.0],
                          [-s_a, c_a,  0.0],
                          [0.0,  0.0,  1.0]])

        bmat = np.matrix([[1.0,  0.0,  0.0],
                          [0.0,  c_b,  s_b],
                          [0.0, -s_b,  c_b]])

        gmat = np.matrix([[c_g,  s_g,  0.0],
                          [-s_g, c_g,  0.0],
                          [0.0,  0.0,  1.0]])

        tmp1 = np.matmul(amat, bmat)
        rmat = np.matmul(tmp1, gmat)

        return rmat

    def combine(self):
        """
        Combines the two galaxies into a new TipsySnap.

        Returns
        ----------
        """
        # these x y z refer to initial condit to be added
        print("Getting initial conditions")
        x1, y1, vx1, vy1 = self._initial_conds(which_gal=self.Gal1)
        x2, y2, vx2, vy2 = self._initial_conds(which_gal=self.Gal2)

        # Mass ratio is unitless so it doesn't matter that they are in kg
        # Units specified and value taken only because later we use in_units
        # Only using simulation units to avoid namespace clutter
        x1 = (x1).to(self.dKpcUnit).value * (self.Mass1_scaled / self.m_tot_scaled)
        y1 = (y1).to(self.dKpcUnit).value * (self.Mass1_scaled / self.m_tot_scaled)
        x2 = (x2).to(self.dKpcUnit).value * (self.Mass2_scaled / self.m_tot_scaled)
        y2 = (y2).to(self.dKpcUnit).value * (self.Mass2_scaled / self.m_tot_scaled)

        vx1 = (vx1).to(self.velUnit).value * (self.Mass1_scaled / self.m_tot_scaled)
        vy1 = (vy1).to(self.velUnit).value * (self.Mass1_scaled / self.m_tot_scaled)
        vx2 = (vx2).to(self.velUnit).value * (self.Mass2_scaled / self.m_tot_scaled)
        vy2 = (vy2).to(self.velUnit).value * (self.Mass2_scaled / self.m_tot_scaled)
        print("Done")
        print("")

        lengths = {}
        for fam in self.Gal1.families():
            lengths[fam.name] = len(self.Gal1[fam])

        gal1_shifted = pynbody.new(**lengths)

        def transform(row, rmat):
            row = (rmat * np.matrix(row).transpose())
            return(row)

        for fam in self.Gal1.families():
            s1 = self.Gal1[fam]
            if self.transform is True:
                print("Tranforming " + str(fam))
                s1["pos"] = np.apply_along_axis(transform, 1, s1["pos"],
                                                (self._rmat(a=self.W1,
                                                            b=self.i1,
                                                            g=self.w1)))

                s1["vel"] = np.apply_along_axis(transform, 1, s1["vel"],
                                                (self._rmat(a=self.W1,
                                                            b=self.i1,
                                                            g=self.w1)))
                print("Done")
                print("")

            else:
                pass

            print("Shifting family " + str(fam))
            # in_units here IS needed to ensure units match when added to IC
            gal1_shifted[fam][:len(s1)]["pos"] = s1["pos"].in_units(str(self.dKpcUnit.value) + " kpc") + [x1, y1, 0]
            gal1_shifted[fam][:len(s1)]["vel"] = s1["vel"].in_units(str(self.velUnit.value * np.sqrt(self.massScale)) + " km s**-1") + [vx1, vy1, 0]
            gal1_shifted[fam][:len(s1)]["mass"] = s1["mass"].in_units(str(self.dMsolUnit.value * self.massScale) + " Msol")
            gal1_shifted[fam][:len(s1)]["rho"] = s1["rho"].in_units(str((self.dMsolUnit * self.massScale / (self.dKpcUnit**3)).value) + " Msol kpc**-3")
            gal1_shifted[fam][:len(s1)]["eps"] = s1["eps"].in_units("kpc")

            if str(fam) == 'gas':
                gal1_shifted[fam][:len(s1)]["temp"] = s1["temp"] * self.massScale
            else:
                pass

            print("Done")
            print("")

        lengths = {}
        for fam in self.Gal2.families():
            lengths[fam.name] = len(self.Gal2[fam])

        gal2_shifted = pynbody.new(**lengths)

        for fam in self.Gal2.families():
            s2 = self.Gal2[fam]
            if self.transform is True:
                print("Tranforming " + str(fam))
                s1["pos"] = np.apply_along_axis(transform, 1, s1["pos"],
                                                (self._rmat(a=self.W2,
                                                            b=self.i2,
                                                            g=self.w2)))

                s1["vel"] = np.apply_along_axis(transform, 1, s1["vel"],
                                                (self._rmat(a=self.W2,
                                                            b=self.i2,
                                                            g=self.w2)))
                print("Done")
                print("")
            else:
                pass

            print("Shifting family " + str(fam))

            gal2_shifted[fam][:len(s2)]["pos"] = s2["pos"].in_units(str(self.dKpcUnit.value) + " kpc") + [x2, y2, 0]
            gal2_shifted[fam][:len(s2)]["vel"] = s2["vel"].in_units(str(self.velUnit.value * np.sqrt(self.massScale)) + " km s**-1") + [vx2, vy2, 0]
            gal2_shifted[fam][:len(s2)]["mass"] = s2["mass"].in_units(str(self.dMsolUnit.value * self.massScale) + " Msol")
            gal2_shifted[fam][:len(s2)]["rho"] = s2["rho"].in_units(str((self.dMsolUnit * self.massScale / (self.dKpcUnit**3)).value) + " Msol kpc**-3")
            gal2_shifted[fam][:len(s2)]["eps"] = s2["eps"].in_units("kpc")

            if str(fam) == 'g':
                gal2_shifted[fam][:len(s2)]["temp"] = s2["temp"] * self.massScale
            else:
                pass

            print("Done")
            print("")

        print("Combining galaxies")
        for fam in gal1_shifted.families():
            lengths[fam.name] = len(gal1_shifted[fam]) + len(gal2_shifted[fam])

        combined = pynbody.new(**lengths)

        for fam in self.Gal1.families():
            s1 = gal1_shifted[fam]
            s2 = gal2_shifted[fam]
            for arname in "pos", "vel", "mass", "rho", "eps":
                combined[fam][:len(s1)][arname] = s1[arname]
                combined[fam][len(s1):][arname] = s2[arname]
            if str(fam) == "gas":
                # temp starts out uniform
                combined[fam][:len(s1)]["temp"] = s1["temp"][:]
                combined[fam][len(s1):]["temp"] = s1["temp"][:]
                print(combined[fam]["temp"])
            else:
                pass

        combined.write(filename=self.writename,
                       fmt=pynbody.tipsy.TipsySnap,
                       cosmological=False)
        print("Done")
        print("")

        return combined

    def _check_gas(self):
        """
        Simply checks whether the simulation will include gas.

        Returns
        ----------
        """
        if (np.sum(self.Gal1.g[:]['mass'])
           and np.sum(self.Gal1.g[:]['mass'])) != 0:
            self.use_gas = True

    def make_param_file(self):
        """
        Creates a param file for use in ChaNGa.
        """
        t = self.get_t_out()
        t_timeunits = t.to(self.timeUnit)
        t_steps = t_timeunits / self.dDelta

        print("t_steps = " + str(t_steps))
        achInFile = self.writename

        # write data in a file.
        file = open(self.writename + ".param", "w")
        L = ["nSteps = " + str(round(t_steps.value)) + "\n",
             "dTheta = 0.7 \n",
             "dEta = .03 \n",
             "dDelta = " + str(self.dDelta) + " \n",
             "iOutInterval = " + str(round(t_steps.value / 5)) + " \n",
             "achInFile = " + achInFile + " \n",
             "achOutName = " + self.writename + " \n",
             "iLogInterval = 1 \n",
             "dMsolUnit = " + str(self.dMsolUnit.value) + " \n",
             "dKpcUnit = " + str(self.dKpcUnit.value) + " \n",
             "dDumpFrameStep = 25 \n",
             "iDirector = 1 \n",
             "bGasAdiabatic = 1"]

        file.writelines(L)
        file.close()

    def make_director_file(self):
        """
        Creates director file for ChaNGa movie output.
        Should set camera to keep entire system in view.
        """
        y = 0.5 * self.inital_separation.value + 20

        self._check_gas()

        if self.use_gas is True:
            L = ["size 1000 1000 \n",
                 "clip 0.001 500 \n",
                 "render tsc \n",
                 "target 0 0 0 \n",
                 "physical \n",
                 "file " + str(self.writename) + " \n",
                 "project ortho \n",
                 "softdark 0.2 \n",
                 "softgas 0.25 \n",
                 "softgassph \n",
                 "logscale 100000000000 10000000000000000 \n",
                 "colstar 1. 0.3 0. 5e-19 \n",
                 "coldark 0.2 0.2 5 6e-18 \n",
                 "colgas 168 54 50 1e-14 \n",
                 "dDumpFrameStep = 25 \n",
                 "iDirector = 1 \n",
                 "FOV 90 \n",
                 "up 0 1 0 \n",
                 "eye 0. 0. " + str(y) + " \n"]
        else:
            L = ["size 1000 1000 \n",
                 "clip 0.001 500 \n",
                 "render tsc \n",
                 "target 0 0 0 \n",
                 "physical \n",
                 "file " + str(self.writename) + " \n",
                 "project perspective \n",
                 "softdark 0.2 \n",
                 "logscale 100000000000 10000000000000000 \n",
                 "colstar 1. 0.3 0. 5e-18 \n",
                 "coldark 0.2 0.2 5 7e-16 \n",
                 "dDumpFrameStep = 25 \n",
                 "iDirector = 1 \n",
                 "gas off \n",
                 "FOV 90 \n",
                 "up 0 1 0 \n",
                 "eye 0. 0. " + str(y) + " \n"]

        file = open(self.writename + ".director", "w")
        file.writelines(L)
        file.close()

    def make_info_file(self):
        """
        Generates a log file for the run with info on galaxy composition.
        """

        self._check_gas()

        IC_mass_dm_1 = np.sum(self.Gal1.dm[:]['mass'])
        IC_mass_s_1 = np.sum(self.Gal1.s[:]['mass'])
        if self.use_gas is True:
            IC_mass_g_1 = np.sum(self.Gal1.g[:]['mass'])
        else:
            IC_mass_g_1 = 0
        IC_totmass_1 = IC_mass_dm_1 + IC_mass_g_1 + IC_mass_s_1

        IC_mass_dm_2 = np.sum(self.Gal2.dm[:]['mass'])
        IC_mass_s_2 = np.sum(self.Gal2.s[:]['mass'])
        if self.use_gas is True:
            IC_mass_g_2 = np.sum(self.Gal2.g[:]['mass'])
        else:
            IC_mass_g_2 = 0
        IC_totmass_2 = IC_mass_dm_2 + IC_mass_g_2 + IC_mass_s_2

        file = open(self.writename + ".info", "w")
        L = ["Writename = self.writename\n",
             "\n",
             "Gal 1 total mass = " + str(self.massScale * IC_totmass_1) + " \n",
             "Gal 1 dm mass = " + str(self.massScale * IC_mass_dm_1) + " \n",
             "Gal 1 star mass = " + str(self.massScale * IC_mass_s_1) + " \n",
             "Gal 1 gas mass = " + str(self.massScale * IC_mass_g_1) + " \n",
             "\n",
             "Gal 2 total mass = " + str(self.massScale * IC_totmass_2) + " \n",
             "Gal 2 dm mass = " + str(self.massScale * IC_mass_dm_2) + " \n",
             "Gal 2 star mass = " + str(self.massScale * IC_mass_s_2) + " \n",
             "Gal 2 gas mass = " + str(self.massScale * IC_mass_g_2) + " \n",
             "\n",
             "Mass unit = " + str(self.dMsolUnit) + " \n",
             "Length unit = " + str(self.dKpcUnit) + " \n",
             "Time unit = " + str(self.timeUnit) + " \n",
             "Velocity unit= " + str(self.velUnit) + " \n"
             ]

        file.writelines(L)
        file.close()

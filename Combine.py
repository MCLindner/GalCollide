import pynbody
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import astropy.units as u
import astropy.constants as const


class GalCombine:
    """
    Create an initial condition for two galaxies in a kepler orbit.
    Attributes:
    Gal1, PyNBody TipsySnap
    Gal2, PyNbody TipsySnap
    d_perigalactic = perigalactic distance given in kpc
    initial_separation = initial separation distance between galaxies in kpc
    eccentriciy = eccentricity of the system
    writename = output file name (string)
    Omega1, w1, i1 = Euler angles for galaxy 1
    Omega2, w2, i2 = Euler angles for galaxy 2
    transform = Bool, whether or not to transform each galaxy by Euler angles

    Public methods:
    combine(): Generates the initial condition tipsy file

    get_period() Gets the period of the orbit of the system

    make_param_file() creates the param file for changa

    make_director_file() creates the director file for changa movie output

    solve_ivp(): tests that the initial condition is
     correct by solving the two body problem.
    """

    def __init__(self, Gal1, Gal2,
                 d_perigalactic, inital_separation, eccentricity,
                 writename, Omega1, w1, i1, Omega2, w2, i2, transform):
        self.Gal1 = Gal1
        self.Gal2 = Gal2
        self.d_perigalactic = d_perigalactic * u.kpc
        self.eccentricity = eccentricity
        self.inital_separation = inital_separation * u.kpc
        self.writename = writename
        self.Omega1 = Omega1
        self.w1 = w1
        self.i1 = i1
        self.Omega2 = Omega2
        self.w2 = w2
        self.i2 = i2
        self.transform = transform

        self.G = const.G
        self.Mass1 = float(self.Gal1['mass'].sum().in_units('kg')) * u.kg
        self.Mass2 = float(self.Gal2['mass'].sum().in_units('kg')) * u.kg
        self.MassTot = self.Mass1 + self.Mass2

    def _get_vxvy(self, which_gal):
        """Finds the velocity componants in cartesian that, given the
        parameters passed into the class, would result
        in a Keplerian two-body orbit."""
        # Constants
        k = np.sqrt(self.G * self.MassTot)

        # Given parameters
        e = self.eccentricity
        rp = self.d_perigalactic
        r = self.inital_separation

        # Derived parameters
        a = -rp / (e - 1)
        b = a * np.sqrt(1 - e**2)
        n = k * a**(-3 / 2)

        # Gal1 ###############################################################
        # Find E1
        def f1(E1):
            return ((np.sqrt((a * (np.cos(E1) - e))**2 + (b * np.sin(E1))**2) - r))
        print("Finding E1")
        E1 = fsolve(f1, x0=0, xtol=10e-6)
        print(E1)
        print("Done")

        # Find positions
        X1 = a * (np.cos(E1) - e)
        Y1 = b * np.sin(E1)

        # Find velocity componants
        vX1 = (-a * n * np.sin(E1)) / (1 - e * np.cos(E1))
        vY1 = (b * n * np.cos(E1)) / (1 - e * np.cos(E1))

        # Transition from position in orbital plane to position in space
        q1 = -1
        x1 = q1 * X1
        y1 = q1 * Y1
        vx1 = q1 * vX1
        vy1 = q1 * vY1

        # Gal2 ###############################################################
        # Find E2
        def f2(E2):
            return (np.sqrt((a * (np.cos(E2) - e))**2 + (b * np.sin(E2))**2) - r)
        print("Finding E2")
        E2 = fsolve(f2, x0=0, xtol=10e-6)
        print(E2)
        print("Done")

        # Find positions
        X2 = a * (np.cos(E2) - e)
        Y2 = b * np.sin(E2)

        # Find velocity componants
        vX2 = (-a * n * np.sin(E2)) / (1 - e * np.cos(E2))
        vY2 = (b * n * np.cos(E2)) / (1 - e * np.cos(E2))

        # Transition from position in orbital plane to position in space
        q2 = 1
        x2 = q2 * X2
        y2 = q2 * Y2
        vx2 = q2 * vX2
        vy2 = q2 * vY2

        # Check that the center of mass is as it should be
        self.xcom = (x1 * self.Mass1 + x2 * self.Mass2) / self.MassTot
        self.ycom = (y1 * self.Mass1 + y2 * self.Mass2) / self.MassTot
        print('xcom = ' + str(self.xcom))

        if which_gal == self.Gal1:
            return x1, y1, vx1, vy1
        elif which_gal == self.Gal2:
            return x2, y2, vx2, vy2

    def get_period(self):
        """Returns the period of orbit in seconds based on Kepler's laws."""
        k = np.sqrt(self.G * self.MassTot)
        a = -self.d_perigalactic / (self.eccentricity - 1)
        T = np.sqrt(((4 * np.pi**2) / (k**2)) * a**3)
        print('T:' + str(T.decompose()))
        return T

    def _initial_conds(self, which_gal):
        """Returns the initial positions and velocities for galaxy given
         by which_gal. Utility function."""
        x, y, vx, vy = self._get_vxvy(which_gal)
        print(str(which_gal) + ': x,y,vx,vy = ' + str((x, y, vx.decompose(), vy.decompose())))
        return x[0], y[0], vx[0], vy[0]

    def _equations(self, t, p):
        """Splits second order diff eqquations of motion into four
         first order differential equations."""
        r = np.sqrt((p[0])**2 + (p[1])**2)

        # Split the diff EQs
        #         u1 = x
        #         u2 = y
        #         u3 = dx
        #         u4 = dy

        du1 = p[2]
        du2 = p[3]
        du3 = -(((self.G * self.MassTot).value) / r**3) * (p[0])
        du4 = -(((self.G * self.MassTot).value) / r**3) * (p[1])
        return [du1, du2, du3, du4]

    def solve_ivp(self, which_gal):
        """Test that the initial condition is correct by
         solving the two-body problem for the system."""
        nsteps = 10000
        period = self.get_period().to(u.s).value
        t_eval = np.linspace(0, period, nsteps)

        if which_gal == self.Gal1:
            x1, y1, vx1, vy1 = self._initial_conds(self.Gal1)
            sol = solve_ivp(self._equations, [0, period],
                            [x1.to(u.m).value, y1.to(u.m).value, vx1.to(u.m/u.s).value, vy1.to(u.m/u.s).value],
                            t_eval=t_eval, rtol=1e-6)
        elif which_gal == self.Gal2:
            x2, y2, vx2, vy2 = self._initial_conds(self.Gal2)
            sol = solve_ivp(self._equations, [0, period],
                            [x2.to(u.m).value, y2.to(u.m).value, vx2.to(u.m/u.s).value, vy2.to(u.m/u.s).value],
                            t_eval=t_eval, rtol=1e-6)
        return sol

    def combine(self):
        """Combines the two galaxies into a new TipsySnap.
         Massunit = 2.2222858e5 Msol. Length unit = 1 kpc."""
        # these x y z refer to initial condit to be added
        print("Getting initial conditions")
        x1, y1, vx1, vy1 = self._initial_conds(which_gal=self.Gal1)
        x2, y2, vx2, vy2 = self._initial_conds(which_gal=self.Gal2)

        x1 = ((x1).to(u.kpc)).value * (self.Mass1 / self.MassTot)
        y1 = ((y1).to(u.kpc)).value * (self.Mass1 / self.MassTot)
        x2 = ((x2).to(u.kpc)).value * (self.Mass2 / self.MassTot)
        y2 = ((y2).to(u.kpc)).value * (self.Mass2 / self.MassTot)

        vx1 = ((vx1).to(.9778 * u.km/u.s)).value * (self.Mass1 / self.MassTot)
        vy1 = ((vy1).to(.9778 * u.km/u.s)).value * (self.Mass1 / self.MassTot)
        vx2 = ((vx2).to(.9778 * u.km/u.s)).value * (self.Mass2 / self.MassTot)
        vy2 = ((vy2).to(.9778 * u.km/u.s)).value * (self.Mass2 / self.MassTot)
        print("Done")

        lengths = {}
        for fam in self.Gal1.families():
            lengths[fam.name] = len(self.Gal1[fam])

        gal1_shifted = pynbody.new(**lengths)

        def transform(row):
            row = (P1 * np.matrix(row).transpose())
            return(row)

        for fam in self.Gal1.families():
            s1 = self.Gal1[fam]
            if self.transform is True:
                Omega1 = self.Omega1
                w1 = self.w1
                i1 = self.i1
                P1 = np.matrix([[np.cos(Omega1)*np.cos(w1)-np.sin(Omega1)*np.cos(i1)*np.sin(w1), -np.cos(Omega1)*np.sin(w1)-np.sin(Omega1)*np.cos(i1)*np.cos(w1), np.sin(Omega1)*np.sin(i1)],
                               [np.sin(Omega1)*np.cos(w1)+np.cos(Omega1)*np.cos(i1)*np.sin(w1), -np.sin(Omega1)*np.sin(w1)+np.cos(Omega1)*np.cos(i1)*np.cos(w1), -np.cos(Omega1)*np.sin(i1)],
                               [np.sin(i1)*np.sin(w1), np.sin(i1)*np.cos(w1), np.cos(i1)]])
                print(P1)
                print("Tranforming " + str(fam))
                s1['pos'] = np.apply_along_axis(transform, 1, s1['pos'])
                s1['vel'] = np.apply_along_axis(transform, 1, s1['vel'])
                print("Done")
            else:
                pass

            print("Shifting family " + str(fam))
            gal1_shifted[fam][:len(s1)]['pos'] = s1['pos'].in_units('kpc') + [x1, y1, 0]
            gal1_shifted[fam][:len(s1)]['vel'] = s1['vel'].in_units('.9778 km s**-1') + [vx1, vy1, 0]
            gal1_shifted[fam][:len(s1)]['mass'] = s1['mass'].in_units('2.2222858e5 Msol')
            gal1_shifted[fam][:len(s1)]['rho'] = s1['rho'].in_units('2.2222858e5 Msol kpc**-3')
            gal1_shifted[fam][:len(s1)]['eps'] = s1['eps'].in_units('kpc')
            print("Done")

        gal1_shifted.write(filename='temp1.tipsy', fmt=pynbody.tipsy.TipsySnap, cosmological=False)

        lengths = {}
        for fam in self.Gal2.families():
            lengths[fam.name] = len(self.Gal2[fam])

        gal2_shifted = pynbody.new(**lengths)

        def transform(row):
            row = (P2 * np.matrix(row).transpose())
            return(row)

        for fam in self.Gal2.families():
            s2 = self.Gal2[fam]
            if self.transform is True:
                Omega1 = self.Omega2
                w1 = self.w1
                i1 = self.i1
                P2 = np.matrix([[np.cos(Omega1)*np.cos(w1)-np.sin(Omega1)*np.cos(i1)*np.sin(w1), -np.cos(Omega1)*np.sin(w1)-np.sin(Omega1)*np.cos(i1)*np.cos(w1), np.sin(Omega1)*np.sin(i1)],
                               [np.sin(Omega1)*np.cos(w1)+np.cos(Omega1)*np.cos(i1)*np.sin(w1), -np.sin(Omega1)*np.sin(w1)+np.cos(Omega1)*np.cos(i1)*np.cos(w1), -np.cos(Omega1)*np.sin(i1)],
                               [np.sin(i1)*np.sin(w1), np.sin(i1)*np.cos(w1), np.cos(i1)]])
                print("Tranforming " + str(fam))
                print(P2)
                s2['pos'] = np.apply_along_axis(transform, 1, s2['pos'])
                s2['vel'] = np.apply_along_axis(transform, 1, s2['vel'])
                print("Done")
            else:
                pass

            print("Shifting family " + str(fam))
            gal2_shifted[fam][:len(s2)]['pos'] = s2['pos'].in_units('kpc') + [x2, y2, 0]
            gal2_shifted[fam][:len(s2)]['vel'] = s2['vel'].in_units('.9778 km s**-1') + [vx2, vy2, 0]
            gal2_shifted[fam][:len(s2)]['mass'] = s2['mass'].in_units('2.2222858e5 Msol')
            gal2_shifted[fam][:len(s2)]['rho'] = s2['rho'].in_units('2.2222858e5 Msol kpc**-3')
            gal2_shifted[fam][:len(s2)]['eps'] = s2['eps'].in_units('kpc')
            print("Done")

        gal2_shifted.write(filename='temp2.tipsy', fmt=pynbody.tipsy.TipsySnap, cosmological=False)

        print("Combining galaxies")
        for fam in gal1_shifted.families():
            lengths[fam.name] = len(gal1_shifted[fam]) + len(gal2_shifted[fam])

        combined = pynbody.new(**lengths)

        for fam in self.Gal1.families():
            s1 = gal1_shifted[fam]
            s2 = gal2_shifted[fam]
            for arname in 'pos', 'vel', 'mass', 'rho', 'eps':
                combined[fam][:len(s1)][arname] = s1[arname]
                combined[fam][len(s1):][arname] = s2[arname]

        combined.write(filename=self.writename, fmt=pynbody.tipsy.TipsySnap, cosmological=False)
        print("Done")
        return combined

    def make_param_file(self):
        """Creates a param file for use in ChaNGa.
         Sets nSteps to integrate over one orbital period."""
        p = self.get_period()

        dDelta = 0.01

        gyears = p.to(1e9 * u.year)
        print(gyears)
        steps = gyears / dDelta

        achInFile = self.writename

        # write data in a file.
        file = open(self.writename + ".param", "w")
        L = ["nSteps = " + str(int(steps.value)) + "\n",
             "dTheta = 0.7 \n",
             "dEta = .03 \n",
             "dDelta = " + str(dDelta) + " \n",
             "iOutInterval = 500 \n",
             "achInFile = " + achInFile + " \n",
             "achOutName = " + self.writename + " \n",
             "iLogInterval = 1 \n",
             "dMsolUnit = 2.2222858e5 \n",
             "dKpcUnit = 1 \n",
             "dDumpFrameStep = 25 \n",
             "iDirector = 1 \n"]

        file.writelines(L)
        file.close()

    def make_director_file(self):
        """Creates director file for ChaNGa movie output.
         Should set camera to keep entire system in view."""
        file = open(self.writename + ".director", "w")
        a = -self.d_perigalactic / (self.eccentricity - 1)
        y = 0.5 * a.value + 20
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

        file.writelines(L)
        file.close()

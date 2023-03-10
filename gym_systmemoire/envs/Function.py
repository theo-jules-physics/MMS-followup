import numpy as np
from numpy import polynomial as poly
from scipy import integrate
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt

def plot_paysage(func, val_lim, x_name, y_name, n=100, marq='.-'):
    # Returns the landscape of func between the limit values val_lim with n points.
    # x_name and y_name give the name of the x and y axes. marq is the marker type.
    x_min, x_max = val_lim
    x = np.linspace(x_min, x_max, n)
    y = func(x)
    figure, ax = belleFigure(x_name, y_name)
    plt.plot(x, y, marq)
    return figure, ax


def solve_EQD(func, X0, t_simu):
    # Numerically solve the differential equation "func" on an array of time "t_simu"
    # for initial conditions "X0". Returns the matrix "sol".
    X0 = np.asarray(X0)
    solver = integrate.ode(func).set_integrator('dopri5')
    solver.set_initial_value(X0)
    sol = np.zeros((len(t_simu), len(X0)))
    for ind in range(len(sol)):
        sol[ind, :] = solver.integrate(t_simu[ind])
    return sol

def solve_EQD_2(func, X0, t_simu):
    X0 = np.asarray(X0)
    sol = np.zeros((len(t_simu), len(X0)))
    sol = odeint_adjoint(func, X0, t_simu, method='dopri5')
    return sol



class Ressort():

    def __init__(self, k, x_e, extr_x, extr_v):
        # init with k the spring stiffness et x_e the spring lengths at the equilibrium positions
        # polynomial is a numpy class
        self.k = k
        self.x_e = np.asarray(x_e)
        self.extr_x = np.asarray(extr_x)
        self.extr_v = np.asarray(extr_v)
        self.__get_x_s()
        self.coeff = poly.polynomial.polyfromroots(x_e)
        self.polyn_force = poly.polynomial.Polynomial(self.coeff)
        self.polyn_energy = self.polyn_force.integ()
        
    def __get_x_s(self):
        nb_pos_stable = int((len(self.x_e)-1)/2)
        self.x_s = [self.x_e[2*k] for k in range(nb_pos_stable)]

    def force(self, x):
        # Returns the force associated with a spring of length x
        return -self.k*self.polyn_force(x)

    def find_ext_force(self):
        # Returns the two extremal forces of a spring
        deriv_poly = self.polyn_force.deriv()
        return deriv_poly.roots()

    def paysage_force(self, val_lim, n=100, marq='.-'):
        # Returns the force landscape
        return plot_paysage(self.force, val_lim, 'Pos', 'Force', n=n, marq=marq)

    def paysage_energy(self, val_lim, n=100, marq='.-'):
        # Returns the energy landscape
        return plot_paysage(self.k*self.polyn_energy, val_lim, 'Pos', 'Energy', n=n, marq=marq)


class EQD_mouvement():
#Can change the app point of the force
    def __init__(self, ressorts, masse, c_frot, change_app_point=False, nb_of_steps=None, threshold=None):
        # init with ressort a list of spring, masse a list of masses between the springs
        # c_frot a list of friction coefficients for every mass
        # X0 the initial conditions.
        self.ressorts = ressorts
        self.masse = np.asarray(masse)
        self.c_frot = np.asarray(c_frot)
        self.change_app_point = change_app_point
        self.nb_of_steps = nb_of_steps
        self.threshold = threshold
        self.n = len(masse)
        self.X_sol = None
        self.t_sol = None

    def force_comp(self, x, force):
        # Compute the forces F of the springs for masses at position x
        dx = np.diff(np.append(0, x))
        F = np.asarray([self.ressorts[k].force(dx[k]) for k in range(len(dx))])
        if self.change_app_point==True:
            assert self.nb_of_steps is not None
            assert self.threshold is not None
            if self.nb_of_steps > self.threshold:
                force_array = np.zeros(self.n)
                force_array[0] = force
                DF = np.append(-np.diff(F), F[-1]) + force_array
            else:
                DF = np.append(-np.diff(F), force + F[-1])
        else:
            DF = np.append(-np.diff(F), force + F[-1])
        return F, DF

    #other way to compute the friction term
    def get_diss_arg(self, v):
        elogations_vel = np.diff(np.append(0, v))
        diss_arg = np.diff(elogations_vel)
        diss_arg = np.append(-diss_arg, elogations_vel[-1])
        return diss_arg

    def solve_EQM(self, force, X0, t_max, n_p=1000):
        # Numerical resolution of EQM for every mass
        self.t_sol = np.linspace(0, t_max, n_p)

        def EQM(t, X):
            x = X[:self.n]
            v = X[self.n:]
            _, DF = self.force_comp(x, force)
            dv = 1/self.masse*(DF-self.c_frot*v)
            return np.append(v, dv)
        self.X_sol = np.transpose(solve_EQD(EQM, X0, self.t_sol))

    def plot_evol_pos(self):
        figure, ax = belleFigure('time', 'Pos')
        for k, x in enumerate(self.X_sol[:self.n, :]):
            plt.plot(self.t_sol, x, '.-', label='Masse n° {}'.format(k))
        plt.legend()

    def plot_evol_diff_pos(self):
        dX_sol = np.diff(np.transpose(np.insert(self.X_sol[:self.n, :], 0, 0, axis=0)))
        figure, ax = belleFigure('time', 'DPos')
        for k, x in enumerate(np.transpose(dX_sol)):
            plt.plot(self.t_sol, x, '.-', label='Masse n° {}'.format(k))
        plt.legend()

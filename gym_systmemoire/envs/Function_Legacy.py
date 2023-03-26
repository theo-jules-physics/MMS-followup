import numpy as np
from numpy import polynomial as poly
from scipy import integrate
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt

def solve_ODE(func, X0, t_simu):
    """
    Numerically solve the differential equation "func" on an array of time "t_simu"
    for initial conditions "X0". Returns the matrix "sol".
    
    Parameters
    ----------
    func : callable
        Differential equation to solve.
    X0 : array_like
        Initial conditions.
    t_simu : array_like
        Array of simulation times.

    Returns
    -------
    sol : ndarray
        Solution matrix.
    """
    X0 = np.asarray(X0)
    solver = integrate.ode(func).set_integrator('dopri5')
    solver.set_initial_value(X0)
    sol = np.zeros((len(t_simu), len(X0)))
    for ind in range(len(sol)):
        sol[ind, :] = solver.integrate(t_simu[ind])
    return sol

class Spring():
    
    def __init__(self, k, x_e, extr_x, extr_v):
        """
        Initialize a Spring object.
        
        Parameters
        ----------
        k : float
            Spring stiffness.
        x_e : array-like
            Spring lengths at the equilibrium positions.
        extr_x : array-like
            Extremal positions.
        extr_v : array-like
            Extremal velocities.
        """
        self.k = k
        self.x_e = np.asarray(x_e)
        self.__get_x_s()
        self.extr_x = np.asarray(extr_x)
        self.extr_v = np.asarray(extr_v)
        self.coeff = poly.polynomial.polyfromroots(x_e)
        self.polyn_force = poly.polynomial.Polynomial(self.coeff)
        
    def __get_x_s(self):
        nb_pos_stable = 1 + int((len(self.x_e)-1)/2)
        self.x_s = [self.x_e[2*k] for k in range(nb_pos_stable)]

    def force(self, x):
        """
        Compute the force associated with a spring of length x.
        
        Parameters
        ----------
        x : float
            Spring length.

        Returns
        -------
        float
            Force value.
        """
        return -self.k*self.polyn_force(x)

    def find_ext_force(self):
        """
        Compute the two extremal forces of a spring.
        
        Returns
        -------
        array-like
            Two extremal forces.
        """
        deriv_poly = self.polyn_force.deriv()
        return deriv_poly.roots()


class ODE_mouvement():
    """Simulates the motion of masses and springs with friction and external force.

    Attributes
    ----------
    springs : list
        List of Spring instances.
    masse : ndarray
        Array of masses between the springs.
    c_frot : ndarray
        Array of friction coefficients for each mass.
    nb_of_steps : int, optional
        Number of steps for the numerical solver.
    threshold : float, optional
        Threshold value for convergence in the solver.
    n : int
        Number of masses.
    X_sol : ndarray
        Solution matrix of positions.
    t_sol : ndarray
        Array of simulation times.
    """

    def __init__(self, springs, masse, c_frot, nb_of_steps=None, threshold=None):
        """Initialize the EQD_mouvement class with given parameters.

        Parameters
        ----------
        springs : list
            List of Spring instances.
        masse : array_like
            List of masses between the springs.
        c_frot : array_like
            List of friction coefficients for every mass.
        nb_of_steps : int, optional
            Number of steps for the numerical solver.
        threshold : float, optional
            Threshold value for convergence in the solver.
        """
        self.springs = springs
        self.masse = np.asarray(masse)
        self.c_frot = np.asarray(c_frot)
        self.nb_of_steps = nb_of_steps
        self.threshold = threshold
        self.n = len(masse)
        self.X_sol = None
        self.t_sol = None

    def force_comp(self, x, force):
        """Compute the forces F of the springs for masses at position x.

        Parameters
        ----------
        x : ndarray
            Array of mass positions.
        force : float
            External force.

        Returns
        -------
        F : ndarray
            Array of spring forces.
        DF : ndarray
            Array of force differences.
        """
        dx = np.diff(np.append(0, x))
        F = np.asarray([self.springs[k].force(dx[k]) for k in range(len(dx))])
        DF = np.append(-np.diff(F), force + F[-1])
        return F, DF

    def get_diss_arg(self, v):
        """Compute the friction term for a given velocity vector.

        Parameters
        ----------
        v : ndarray
            Array of velocities.

        Returns
        -------
        diss_arg : ndarray
            Array of friction term values.
        """
        elogations_vel = np.diff(np.append(0, v))
        diss_arg = np.diff(elogations_vel)
        diss_arg = np.append(-diss_arg, elogations_vel[-1])
        return diss_arg

    def solve_ODE(self, force, X0, t_max, n_p=1000):
        """
        Numerically solve the equation of motion for each mass.

        Parameters
        ----------
        force : float
            External force applied.
        X0 : array_like
            Initial conditions.
        t_max : float
            Maximum simulation time.
        n_p : int, optional
            Number of time points, default is 1000.
        """
        self.t_sol = np.linspace(0, t_max, n_p)

        def ODE(t, X):
            x = X[:self.n]
            v = X[self.n:]
            _, DF = self.force_comp(x, force)
            dv = 1 / self.masse * (DF - self.c_frot * v)
            return np.append(v, dv)

        self.X_sol = np.transpose(solve_ODE(ODE, X0, self.t_sol))

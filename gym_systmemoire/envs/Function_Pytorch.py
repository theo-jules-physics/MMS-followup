from numpy import polynomial as poly
import torch
from torchdiffeq import odeint

def solve_ODE(func, X0, t_simu, device='cpu'):
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
    X0 = torch.as_tensor(X0, device=device, dtype=torch.float64)
    t = torch.as_tensor(t_simu, device=device, dtype=torch.float64)
    sol = odeint(func, X0, t)
    return sol

class Spring():
    
    def __init__(self, k, x_e, extr_x, extr_v, device='cpu'):
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
        self.device = device
        self.x_e = torch.as_tensor(x_e, device=self.device)
        self._get_x_s()
        self.extr_x = torch.as_tensor(extr_x, device=self.device)
        self.extr_v = torch.as_tensor(extr_v, device=self.device)    
        self.coeff = poly.polynomial.polyfromroots(x_e)
        self.polyn_force = poly.polynomial.Polynomial(self.coeff)
        
        
    def _get_x_s(self):
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

    def __init__(self, springs, masse, c_frot, nb_of_steps=None, threshold=None,
                 device='cpu'):
        """Initialize the EQD_mouvement class with given parameters.

        Parameters
        ----------
        springs : list
            List of Spring instances.
        masse : array_like
            List of masses between the springs.
        c_frot : array_like
            List of friction coefficients for every mass.
        """
        self.springs = springs
        self.device = device
        self.masse = torch.as_tensor(masse, device=self.device)
        self.c_frot = torch.as_tensor(c_frot, device=self.device)
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
        dx = torch.diff(torch.cat([torch.tensor([0], device=self.device), x]))
        F = torch.stack([self.springs[k].force(dx[k]) for k in range(len(dx))])
        DF = torch.cat((-torch.diff(F), (force + F[-1]).reshape(1)))
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
        elogations_vel = torch.diff(torch.cat([torch.tensor([0], device=self.device), v]))
        diss_arg = torch.diff(elogations_vel)
        diss_arg = torch.cat((-diss_arg, elogations_vel[-1].unsqueeze(0)), dim=0)
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
        self.t_sol = torch.linspace(0, t_max, n_p, device=self.device)

        def ODE(t, X):
            x = X[:self.n]
            v = X[self.n:]
            _, DF = self.force_comp(x, force)
            dv = 1 / self.masse * (DF - self.c_frot * v)
            return torch.cat((v, dv), dim=0)

        self.X_sol = torch.transpose(solve_ODE(ODE, X0, self.t_sol, device=self.device), 0, 1)

import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random

from gym_systmemoire.envs import Function

import copy

metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
}

class Couplage_n_Env_Spring(gym.Env):   
    
    def __init__(self, masse, max_f, system, c, dt,
                 limit_reset=[0.2, 0.1], goal_state=None, recup_traj=True, final_state=None,
                 start_from_eq_pos=False, start_with_gaussian_vel=False, add_gaussian_noise_to_ini_pos=False, pos_noise=0.001, vel_noise=0.01,
                 change_app_point=False, threshold=1000000, cond_success=[0.005, 0.01], reward_success=50, pen_coeff=[1, 0.5]):

        self.viewer = None
        self.max_f = max_f
        self.system = system
        self.nb_ressort = np.size(system)
        self.recup_traj = recup_traj
        self.start_from_eq_pos = start_from_eq_pos
        self.start_with_gaussian_vel = start_with_gaussian_vel
        self.add_gaussian_noise_to_ini_pos = add_gaussian_noise_to_ini_pos
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.change_app_point = change_app_point
        self.threshold = threshold
        self.reward_success = reward_success
        self.limit_reset = limit_reset
        self.cond_success = cond_success
        self.pen_coeff = pen_coeff
        self.c = c
        self.dt = dt  # 0.1
        self.goal_state = goal_state
        self.nb_mass_learn = self.nb_ressort
        self.masse = masse
        self.final_state = final_state

        self.x, self.extr_x, self.extr_v, self.k = self.__recup_info_syst()

        self.low_state = np.concatenate((self.extr_x[0, :],
                                         self.extr_v[0, :],
                                         [0 for k in range(self.nb_ressort)])).astype(np.float32)
        self.high_state = np.concatenate((self.extr_x[1, :],
                                          self.extr_v[1, :],
                                          [1 for k in range(self.nb_ressort)])).astype(np.float32)
        self.action_space = spaces.Box(low=-self.max_f, high=self.max_f, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.state = None
        self.goal = None
        self.traj = dict(positions=[], velocities=[], action=[], reward=[])
        self.reward_episode = []
        self.reset()
        self.nb_of_steps = 0
        
        
        self.__seed() 

    def __seed(self, seed=None):
        """
        Set the random seed

        Parameters
        ----------
        seed : int, optional
            Value of the random seed. The default is None.

        Returns
        -------
        seed : int
            Value of the random seed.
        """

        self.np_random, seed = seeding.np_random(seed)
        return seed
        """
        Regroup the static data from the springs in the system:
        Eq. position, Extreme values for positions/velocities, Springs stiffness.
        
        Returns
        -------
        x : np.array
            Eq. position for all the springs.
        extr_x : np.array
            Extreme values for positions.
        extr_v : np.array
            Extreme values for velocities.
        k : np.array
            Springs stiffness.
        """
        x = np.zeros((self.nb_pos_eq, self.nb_ressort))
        extr_x = np.zeros((2, self.nb_ressort))
        extr_v = np.zeros((2, self.nb_ressort))
        k = np.zeros(self.nb_ressort)
        for i in range(self.nb_ressort):
            x[:, i] = copy.copy(self.system[i].x_e)
            extr_x[:, i] = copy.copy(self.system[i].extr_x)
            extr_v[:, i] = copy.copy(self.system[i].extr_v)
            k[i] = copy.copy(self.system[i].k)
        return x, extr_x, extr_v, k

    def __update_traj(self, x, v, f, reward):
        """
        Add the state of the system + reward + action to the trajectory.
        
        Parameters
        ----------
        x : list
            Positions of the masses.
        v : list
            Velocities of the masses.
        f : float
            Force applied to the last mass.
        reward : float
            Corresponding reward.
        Returns
        -------
        None.
        """
        self.traj['positions'].append(x)
        self.traj['velocities'].append(v)
        self.traj['action'].append(f)
        self.traj['reward'].append(reward)
        
    def __recup_info_syst(self):
        nb_pos_eq = len(self.system[0].x_e)
        x = np.zeros((nb_pos_eq, self.nb_ressort))
        extr_x = np.zeros((2, self.nb_ressort))
        extr_v = np.zeros((2, self.nb_ressort))
        k = np.zeros(self.nb_ressort)

        for i in range(self.nb_ressort):
            x[:, i] = copy.copy(self.system[i].x_e)
            extr_x[:, i] = copy.copy(self.system[i].extr_x)
            extr_v[:, i] = copy.copy(self.system[i].extr_v)
            k[i] = copy.copy(self.system[i].k)

        return x, extr_x, extr_v, k
    
    def _get_ob(self):
        s = self.state
        return s[:]

    def bin_to_pos(self, binstate, eq_pos):
        """
        Transform the number of the targeted state to a specific equilibrium position.

        Parameters
        ----------
        binstate : int
            Number of the targeted stable position.
        eq_pos : list
            List of all the equilibrium positions.

        Returns
        -------
        Float
            Equilibrium position for the spring.

        """
        return eq_pos[int(2*binstate+1)]

    def goalbin_to_goalpos(self):
        """
        Get the list of positions corresponding to a specified target state.

        Returns
        -------
        goalpos : list
            List of stable positions.

        """
        goalpos = np.zeros_like(self.goal, dtype='float')
        for i in range(self.nb_ressort):
            goalpos[i] = self.bin_to_pos(self.goal[i], self.x[:, i])
        return goalpos

    def compute_error(self, rel_pos, goalpos):
        """
        Compute the absolute difference in positions and velocities between the
        actual state of the springs and the targeted state.

        Parameters
        ----------
        rel_pos : list
            Relative position of the springs.
        goalpos : list
            Targeted positions.

        Returns
        -------
        list
            Absolute error for the positions and the velocities.
        """
        pos_error = np.zeros(self.nb_ressort)
        vel_error = np.zeros(self.nb_ressort)
        
        for i in range(self.nb_ressort):
            pos_error[i] = np.abs(rel_pos[i] - goalpos[i])
            vel_error[i] = np.abs(self.state[self.nb_ressort + i])

        return [pos_error, vel_error]
    
    def isterminal(self, errors):
        """
        Check if the system reached the end of the epoch by completing the goal

        Parameters
        ----------
        errors : list
            Absolute error for the positions and the velocities.

        Returns
        -------
        isterminal : bool
            Boolean indicating if the goal is accomplished.
        """
        pos_error, vel_error = errors
        verif_pos = np.zeros(self.nb_mass_learn)
        verif_vel = np.zeros(self.nb_mass_learn)
        cond_pos, cond_vel = self.cond_success
        for i in range(self.nb_mass_learn):
            verif_pos[i] = pos_error[-1-i] < cond_pos
            verif_vel[i] = vel_error[-1-i] < cond_vel
        if np.sum(verif_pos + verif_vel) == 2*self.nb_mass_learn:
            isterminal = True
        else:
            isterminal = False
        return isterminal

    def give_reward(self, isterminal, errors):
        """
        Get the reward/penalty given an error depending on whether the goal is
        reached or not.

        Parameters
        ----------
        isterminal : bool
            Boolean indicating if the goal is accomplished..
        errors : List
            Absolute error for the positions and the velocities.

        Returns
        -------
        reward : float
            Reward associated to the step computed.

        """
        pos_error, vel_error = errors
        if isterminal:
            reward = self.reward_success
        else:
            c_pos, c_vel = self.pen_coeff
            reward = - c_pos*np.sum(pos_error) - c_vel*np.sum(vel_error)
        return reward

    def reset(self):
        """
        Reset the system to start a new epoch. This correspond to a reset of the
        positions and velocities of the masses + setting a new target.

        Returns
        -------
        state
            State of the system.

        """
        dx, dv = self.limit_reset
        x_rel, v = [], []
        for k in range(self.nb_ressort):
            v.append(self.np_random.uniform(low=-dv, high=dv))
            if self.start_from_eq_pos==True:
                x_rel.append(np.random.choice(self.system[k].x_s))
            else:
                x_rel.append(self.np_random.uniform(low=self.x[0, k]-dx, high=self.x[-1, k]+dx))
        x = np.cumsum(x_rel)
        x_v = np.concatenate((x, v))
            
        self.goal = []  
        if self.final_state == None:
            for k in range(self.nb_ressort):
                self.goal.append(random.choice([0, 1]))
        else:
            self.goal =self.final_state

        self.state = np.concatenate((x_v, self.goal))

        if self.recup_traj:
            self.traj['positions'] = np.asarray(x)
            self.traj['velocities'] = np.asarray(v)
            self.traj['action'] = []
            errors = self.compute_error(x_rel, self.goalbin_to_goalpos())
            self.traj['reward'] = [self.give_reward(False, errors)]

        return self._get_ob()
    
    def step(self, action):
        """
        Advance the simulation by one step given an action.

        Parameters
        ----------
        action : float
            Action applied at this step.

        Returns
        -------
        state
            State of the system.
        reward : float
            Reward associated with the computed step.
        done : bool
            Indicate whether this step is the last from the epoch.
        dict
            ???.

        """

        self.nb_of_steps += 1

        force = min(max(action[0], -self.max_f), self.max_f)

        eqd_syst = Function.EQD_mouvement(self.system, self.masse, self.c * np.ones(self.nb_ressort), self.change_app_point,
                                          self.nb_of_steps, self.threshold)

        s = self.state[: -self.nb_ressort]
        eqd_syst.solve_EQM(force, s, self.dt, n_p=10)
        sol = eqd_syst.X_sol[:, -1]
        rel_pos = np.diff(sol[:self.nb_ressort], prepend=0)
        self.state = np.concatenate((sol, self.goal))

        goalpos = self.goalbin_to_goalpos()
        errors = self.compute_error(rel_pos, goalpos)
        done = self.isterminal(errors)
        reward = self.give_reward(done, errors)

        if self.recup_traj:
            self.__update_traj(sol[:self.nb_ressort], sol[self.nb_ressort:], force, reward)

        return (self._get_ob(), reward, done, {})

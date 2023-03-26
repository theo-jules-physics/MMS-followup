import torch
import gym
from gym import spaces
import random

from gym_systmemoire.envs import Function

import copy

class Couplage_n_Env_Spring(gym.Env):   
    
    def __init__(self, masse, max_f, system, c, dt, limit_reset=[0.2, 0.1], 
                 goal_state=None, recup_traj=True, final_state=None,
                 start_from_eq_pos=False, threshold=1000000, cond_success=[0.005, 0.01],
                 reward_success=50, pen_coeff=[1, 0.5], device='cpu'):

        self.device = device
        self.max_f = max_f
        self.system = system
        self.nb_springs = len(system)
        self.recup_traj = recup_traj
        self.start_from_eq_pos = start_from_eq_pos
        self.threshold = threshold
        self.reward_success = reward_success
        self.limit_reset = limit_reset
        self.cond_success = cond_success
        self.pen_coeff = pen_coeff
        self.c = c
        self.dt = dt  # 0.1
        self.goal_state = goal_state
        self.masse = masse
        self.final_state = final_state

        self.x, self.extr_x, self.extr_v, self.k = self.__recup_info_syst()

        self.low_state = torch.cat((self.extr_x[0, :], self.extr_v[0, :], 
                                    torch.zeros(self.nb_springs, device=self.device)))
        self.high_state = torch.cat((self.extr_x[1, :], self.extr_v[1, :],
                                     torch.zeros(self.nb_springs, device=self.device)))
        self.action_space = spaces.Box(low=-self.max_f, high=self.max_f, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state.cpu().numpy(), high=self.high_state.cpu().numpy())

        self.state = None
        self.goal = None
        self.traj = dict(positions=[], velocities=[], action=[], reward=[])
        self.reward_episode = []
        self.reset()
        self.nb_of_steps = 0

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
        x = torch.zeros((nb_pos_eq, self.nb_springs), device=self.device)
        extr_x = torch.zeros((2, self.nb_springs), device=self.device)
        extr_v = torch.zeros((2, self.nb_springs), device=self.device)
        k = torch.zeros(self.nb_springs, device=self.device)

        for i in range(self.nb_springs):
            x[:, i] = copy.copy(self.system[i].x_e)
            extr_x[:, i] = copy.copy(self.system[i].extr_x)
            extr_v[:, i] = copy.copy(self.system[i].extr_v)
            k[i] = copy.copy(self.system[i].k)

        return x, extr_x, extr_v, k
    
    def _get_ob(self):
        s = self.state
        return s[:]

    def goalbin_to_goalpos(self):
        """
        Get the list of positions corresponding to a specified target state.

        Returns
        -------
        goalpos : list
            List of stable positions.

        """
        goalpos = torch.zeros_like(self.goal, device=self.device)
        for i in range(self.nb_springs):
            goalpos[i] = self.system[i].x_s[self.goal[i]]
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
        pos_error = torch.zeros(self.nb_springs, device=self.device)
        vel_error = torch.zeros(self.nb_springs, device=self.device)
        
        for i in range(self.nb_springs):
            pos_error[i] = torch.abs(rel_pos[i] - goalpos[i])
            vel_error[i] = torch.abs(self.state[self.nb_springs + i])

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
        verif_pos = torch.zeros(self.nb_springs, device=self.device)
        verif_vel = torch.zeros(self.nb_springs, device=self.device)
        cond_pos, cond_vel = self.cond_success
        for i in range(self.nb_springs):
            verif_pos[i] = pos_error[-1-i] < cond_pos
            verif_vel[i] = vel_error[-1-i] < cond_vel
        if torch.sum(verif_pos + verif_vel) == 2*self.nb_springs:
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
            reward = - c_pos*torch.sum(pos_error) - c_vel*torch.sum(vel_error)
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
        for k in range(self.nb_springs):
            v.append(self.np_random.uniform(low=-dv, high=dv))
            x_min , x_max = self.x[0, k]-dx, self.x[-1, k]+dx
            x_rel.append(x_min + torch.rand(1, device=self.device)*(x_max-x_min))
        x = torch.cumsum(torch.as_tensor(x_rel, device=self.device), 0)
        x_v = torch.cat((x, torch.as_tensor(v, device=self.device)))
            
        self.goal = []  
        if self.final_state == None:
            for k in range(self.nb_springs):
                self.goal.append(random.choice([0, 1]))
        else:
            self.goal =self.final_state
        self.goal = torch.as_tensor(self.goal, device=self.device)
        self.state = torch.cat((x_v, self.goal))

        if self.recup_traj:
            errors = self.compute_error(x_rel, self.goalbin_to_goalpos())
            reward = self.give_reward(False, errors)
            self.__update_traj(x, v, None, reward)

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

        eqd_syst = Function.ODE_mouvement(self.system, self.masse, self.c * torch.ones(self.nb_springs, device=self.device),
                                          self.nb_of_steps, self.threshold, device=self.device)

        s = self.state[: -self.nb_springs]
        eqd_syst.solve_ODE(force, s, self.dt, n_p=10)
        sol = eqd_syst.X_sol[:, -1]
        rel_pos = torch.diff(sol[:self.nb_springs], prepend=torch.tensor(0, device=self.device).reshape(1))
        self.state = torch.cat((sol, self.goal))

        goalpos = self.goalbin_to_goalpos()
        errors = self.compute_error(rel_pos, goalpos)
        done = self.isterminal(errors)
        reward = self.give_reward(done, errors)

        if self.recup_traj:
            self.__update_traj(sol[:self.nb_springs], sol[self.nb_springs:], force, reward)

        return (self._get_ob(), reward, done, {})

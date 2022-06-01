import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random

from gym_systmemoire.envs import Function

import copy


class Couplage_n_Env_Spring(gym.Env):
    """
        This environment describes a one-dimensional multistable chain composed of coupled bistable spring-mass units.
        The first spring of the chain is attached to a fixed wall and an external force is applied at the end of the chain.
        The aim is to put the system in a chosen equilibrium configuration by choosing a sequence of external forces.

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, masse, max_f, system, nb_pos_eq, c, dt, ini_pos=None,
                 limit_reset=[0.2, 0.1], goal_state=None, recup_traj=True,
                 start_from_eq_pos=False, start_with_gaussian_vel=False, add_gaussian_noise_to_ini_pos=False, pos_noise=0.001, vel_noise=0.01,
                 change_app_point=False, threshold=1000000, cond_success=[0.005, 0.01], reward_sucess=50, pen_coeff=[1, 0.5]):
        
        self.viewer = None
        self.max_f = max_f
        self.system = system
        self.nb_ressort = np.size(system)
        self.nb_pos_eq = nb_pos_eq
        self.recup_traj = recup_traj
        self.start_from_eq_pos = start_from_eq_pos
        self.start_with_gaussian_vel = start_with_gaussian_vel
        self.add_gaussian_noise_to_ini_pos = add_gaussian_noise_to_ini_pos
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.change_app_point = change_app_point
        self.threshold = threshold
        self.reward_sucess = reward_sucess
        self.limit_reset = limit_reset
        self.cond_success = cond_success
        self.pen_coeff = pen_coeff
        self.c = c
        self.dt = dt  # 0.1
        self.goal_state = goal_state
        self.ini_pos = ini_pos
        self.nb_mass_learn = self.nb_ressort
        self.masse = masse

        self.x, self.extr_x, self.extr_v, self.k = self.__recup_info_syst()

        self.low_state = np.concatenate((self.extr_x[0, :],
                                         self.extr_v[0, :],
                                         [0 for k in range(self.nb_ressort)])).astype(np.float32)
        self.high_state = np.concatenate((self.extr_x[1, :],
                                          self.extr_v[1, :],
                                          [1 for k in range(self.nb_ressort)])).astype(np.float32)
        self.action_space = spaces.Box(low=-self.max_f, high=self.max_f, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.ep_is_beginning = False

        self.state = None
        self.goal = None
        self.traj_x = None
        self.traj_v = None
        self.traj_f = None
        self.traj_reward = None

        self.reward_episode = []
        self.nb_of_steps = 0

        
        self.__seed()


    ###--------------------------------------------###
    def __recup_info_syst(self):

        x = np.zeros((self.nb_pos_eq, self.nb_ressort))
        extr_x = np.zeros((2, self.nb_ressort))
        extr_v = np.zeros((2, self.nb_ressort))
        k = np.zeros(self.nb_ressort)

        for i in range(self.nb_ressort):
            x[:, i] = copy.copy(self.system[i].x_e)
            extr_x[:, i] = copy.copy(self.system[i].extr_x)
            extr_v[:, i] = copy.copy(self.system[i].extr_v)

        return x, extr_x, extr_v, k

    ###--------------------------------------------###
    def __update_traj(self, x, v, f, reward):
        x = np.asarray(x)
        self.traj_x = np.vstack((self.traj_x, x))
        self.traj_v = np.vstack((self.traj_v, v))
        self.traj_f.append(f)
        self.traj_reward.append(reward)

    ###--------------------------------------------###
    def __update_reward_episode(self, reward):
        self.reward_episode.append(reward)

    ###--------------------------------------------###
    def __seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ###--------------------------------------------###
    def _get_ob(self):
        s = self.state
        return s[:]

    ###--------------------------------------------###
    def bin_to_pos(self, binstate, eq_pos):
        if binstate == 0:
            become = eq_pos[0]
        else:
            become = eq_pos[2]
        return become

    ###--------------------------------------------###
    def goalbin_to_goalpos(self):
        goalpos = np.zeros_like(self.goal, dtype='float')
        for i in range(self.nb_ressort):
            goalpos[i] = self.bin_to_pos(self.goal[i], self.x[:, i])
        return goalpos

    ###--------------------------------------------###
    def compute_error(self, rel_pos, goalpos):
        pos_error = np.zeros(self.nb_ressort)
        vel_error = np.zeros(self.nb_ressort)
        
        for i in range(self.nb_ressort):
            pos_error[i] = np.abs(rel_pos[i] - goalpos[i])
            vel_error[i] = np.abs(self.state[self.nb_ressort + i])

        return [pos_error, vel_error]
    
    ###--------------------------------------------###
    def isterminal(self, errors):
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

    ###--------------------------------------------###
    def give_reward(self, isterminal, errors):
        pos_error, vel_error = errors
        
        if isterminal:
            reward = self.reward_sucess
        else:
            c_pos, c_vel = self.pen_coeff
            reward = - c_pos*np.sum(pos_error) - c_vel*np.sum(vel_error)
        return reward

    ###--------------------------------------------###
    def reset(self):
        self.ep_is_beginning = True
        if np.size(self.reward_episode) != 0:
            self.__update_reward_episode(self.traj_reward[-1])
        if self.ini_pos == None:
            dx, dv = self.limit_reset
            x0_relatif = [self.np_random.uniform(low=self.x[0, k], high=self.x[-1, k]+dx) for k in range(self.nb_ressort)]
            v =[self.np_random.uniform(low=-dv, high=dv) for k in range(self.nb_ressort)]

            if self.start_from_eq_pos==True:
                x0_relatif = np.zeros_like(x0_relatif)
                for i in range(self.nb_ressort):
                    x0_relatif[i] = np.random.choice([self.system[i].x_e[0], self.system[i].x_e[2]]) + np.random.normal(scale=self.pos_noise)

            x = np.zeros_like(x0_relatif)
            cumulative_pos = 0
            for i in range(self.nb_ressort):
                x[i] = cumulative_pos + x0_relatif[i]
                cumulative_pos += x0_relatif[i]

            x_v = np.concatenate((x, v))

        else:
            x_v = copy.copy(self.ini_pos)

        if self.start_with_gaussian_vel==True:
            for i in range(self.nb_ressort):
                x_v[self.nb_ressort + i] = np.random.normal(0., self.vel_noise, 1)[0]

        if self.add_gaussian_noise_to_ini_pos == True:
            for i in range(self.nb_ressort):
                x_v[i] = x_v[i] + np.random.normal(scale=self.pos_noise)


        self.goal = []        
        for final_state in self.goal_state:
            if final_state == None:
                self.goal.append(random.choice([0, 1]))
            else:
                self.goal.append(final_state)

        self.state = np.concatenate((x_v, self.goal))

        if self.recup_traj:
            self.traj_x = np.asarray(x_v[:self.nb_ressort])
            self.traj_v = np.asarray(x_v[self.nb_ressort:])
            self.traj_f = [0]
            rel_pos = np.diff(np.append(0, self.state[:self.nb_ressort]))
            errors = self.compute_error(rel_pos, self.goalbin_to_goalpos())
            self.traj_reward = [self.give_reward(False, errors)]
            if np.size(self.reward_episode) == 0:
                self.__update_reward_episode(0)

        return self._get_ob()

    ###--------------------------------------------###
    def step(self, action):
        ''' n steps in an episode '''

        self.nb_of_steps += 1

        force = min(max(action[0], -self.max_f), self.max_f)

        eqd_syst = Function.EQD_mouvement(self.system, self.masse, self.c * np.ones(self.nb_ressort), self.change_app_point,
                                          self.nb_of_steps, self.threshold)

        s = self.state[: -self.nb_ressort]
        eqd_syst.solve_EQM(force, s, self.dt, n_p=10)
        sol = eqd_syst.X_sol[:, -1]
        rel_pos = np.diff(np.append(0, sol[:self.nb_ressort]))
        self.state = np.concatenate((sol, self.goal))

        goalpos = self.goalbin_to_goalpos()
        errors = self.compute_error(rel_pos, goalpos)
        done = self.isterminal(errors)
        reward = self.give_reward(done, errors)

        if self.recup_traj:
            self.__update_traj(sol[:self.nb_ressort], sol[self.nb_ressort:], force, reward)

        return (self._get_ob(), reward, done, {})

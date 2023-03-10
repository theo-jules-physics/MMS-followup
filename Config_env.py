import numpy as np
import gym_systmemoire.envs.Function
from gym_systmemoire.envs.Function import Ressort

# max_f : extremal force the system can apply on the last mass (+/- max_f)
# system : system configuration with the properties of the different springs
# nb_pos_eq : number of equilibrium position for all the springs
# c : friction coefficient
# dt : timestep between change of the action/force
# ini_pos : set the initial position in an epoch if needed
# limit_reset : List of 2 numbers, a and b, that set the extremal value for the 
# randomly chosen position (between the minimal equilibrium position -a and the 
#                           maximum + a) and velocity (between -b and b) of each
# mass at the beggining of each epoch.
# goal_state : define if the system has specific or random (None) targets.
# recup_traj : set if the trajectory of the masses are recorded for a given fit.
# cond_sucess : set the condition for the sucess of an epoch
# reward_sucess : set the value for the reward in case of sucess
# pen_coeff : coeff for the penalty for the positions and velocities
exp = dict()

######### 1 mass ##########

exp['1M-v0'] = {'masse': np.array([1.]), 'max_f': 1, 'system': [Ressort(88.7, [0.05, 0.100, 0.150], [0, 1.5], [-1, 1])],
                'nb_pos_eq': 3, 'c': 2., 'dt': 0.1, 'ini_pos': None, 'limit_reset': [0.2, 0.1], 'goal_state': [None],
                'recup_traj': True, 'cond_success': [0.005, 0.01], 'reward_sucess': 50, 'pen_coeff': [1, 0.5]}


######### 2 masses ##########

exp['2M-v0'] = {'max_f': 1, 'system': [Ressort(88.7, [0.040, 0.080, 0.100],[0, 1.5], [-1, 1]),
                                       Ressort(88.7, [0.030, 0.060, 0.105],[0, 1.5], [-1, 1])],
                'nb_pos_ eq':3, 'c': 2., 'dt': 0.1, 'ini_pos': None, 'limit_reset': [0.2,0.1], 'goal_state': [None, None],
                'recup_traj': True, 'cond_success': [0.005, 0.01], 'reward_sucess': 50, 'pen_coeff': [1, 0.5]}

######### 3 masses ##########

exp['3M-v0'] = {'masse': np.array([1., 1., 1.]), 'max_f': 1, 'system': [Ressort(88.7, [0.05, 0.100, 0.150], [0, 1.5], [-1, 1]),
                                                                        Ressort(88.7, [0.040, 0.080, 0.100],[0, 1.5], [-1, 1]),
                                                                        Ressort(88.7, [0.030, 0.060, 0.105],[0, 1.5], [-1, 1])],
                'nb_pos_eq': 3, 'c': 2., 'dt': 0.1, 'ini_pos': None, 'limit_reset': [0.2, 0.1], 'goal_state': [None, None, None],
                'recup_traj': True, 'cond_success': [0.005, 0.01],  'reward_sucess': 50, 'pen_coeff': [1, 0.5]}


######### 4 masses ##########

exp['4M-v0'] = {'masse': np.array([1., 1., 1., 1.]), 'max_f': 1, 'system': [Ressort(88.7, [0.05, 0.100, 0.150], [0, 1.5], [-1, 1]),
                                       Ressort(88.7, [0.040, 0.080, 0.100],[0, 1.5], [-1, 1]),
                                       Ressort(88.7, [0.030, 0.060, 0.105],[0, 1.5], [-1, 1]),
                                       Ressort(88.7, [0.055, 0.110, 0.165],[0, 1.5], [-1, 1])],
                'nb_pos_eq': 3, 'c': 2., 'dt': 0.1, 'ini_pos': None, 'limit_reset': [0.2, 0.1], 'goal_state': [None, None, None, None],
                'recup_traj': True, 'cond_success': [0.005, 0.01], 'reward_sucess': 50, 'pen_coeff': [1, 0.5]}

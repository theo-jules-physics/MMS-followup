import numpy as np
import gym_systmemoire.envs.Function
from gym_systmemoire.envs.Function import Spring

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

######### 3 masses ##########

exp['3M'] = {'masse': np.array([1., 1., 1.]), 'max_f': 1, 'system': [Spring(88.7, [0.05, 0.100, 0.150], [0, 1.5], [-1, 1]),
                                                                        Spring(88.7, [0.040, 0.080, 0.100],[0, 1.5], [-1, 1]),
                                                                        Spring(88.7, [0.030, 0.060, 0.105],[0, 1.5], [-1, 1])],
             'c': 2., 'dt': 0.1, 'limit_reset': [0.2, 0.1], 'goal_state': [None, None, None], 'recup_traj': True,
             'cond_success': [0.005, 0.01],  'reward_success': 50, 'pen_coeff': [1, 0.5]}

exp['2M'] = {'masse': np.array([1., 1.]), 'max_f': 1, 'system': [Spring(88.7, [0.05, 0.100, 0.150], [0, 1.5], [-1, 1]),
                                                                        Spring(88.7, [0.040, 0.080, 0.100],[0, 1.5], [-1, 1])],
             'c': 2., 'dt': 0.1, 'limit_reset': [0.2, 0.1], 'goal_state': [None, None, None], 'recup_traj': True,
             'cond_success': [0.005, 0.01],  'reward_success': 50, 'pen_coeff': [1, 0.5]}

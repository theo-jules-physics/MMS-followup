import numpy as np
import logging
import os
import torch
from torch import nn
import gym
import gym_systmemoire
import gym_systmemoire.envs
import pandas as pd
import plotly.graph_objects as go
import Config_env

assert torch.cuda.is_available()
device = torch.device('cpu')

class Train():
    def __init__(self, seed, put_seed, gpu, env_name, gamma, rb_size, tau, replay_start_size, minibatch_size, steps, eval_n_episodes,
                 eval_interval, train_max_episode_len, path_to_save, path_for_loading, dirname_to_save, dirname_for_loading, tar_act_noise, threshold=None, noise=None, scheduler=None,
                 automatically_stop=False, success_threshold=0.95, save_force_signal=False, wrong_keys=True, nb_of_neurons_layer_1=400, nb_of_neurons_layer_2=300):
        self.seed = seed
        self.put_seed = put_seed
        self.gpu = gpu
        self.env_name = env_name
        self.gamma = gamma
        self.rb_size = rb_size
        self.tau = tau
        self.replay_start_size = replay_start_size
        self.minibatch_size = minibatch_size
        self.steps = steps
        self.eval_n_episodes = eval_n_episodes
        self.eval_interval = eval_interval
        self.train_max_episode_len = train_max_episode_len
        self.path_to_save = path_to_save
        self.path_for_loading = path_for_loading
        self.dirname_to_save = dirname_to_save
        self.dirname_for_loading = dirname_for_loading
        self.tar_act_noise = tar_act_noise
        self.threshold = threshold
        self.noise = noise
        self.scheduler = scheduler
        self.automatically_stop = automatically_stop
        self.success_threshold = success_threshold
        self.save_force_signal = save_force_signal
        self.wrong_keys = wrong_keys
        self.nb_of_neurons_layer_1 = nb_of_neurons_layer_1
        self.nb_of_neurons_layer_2 = nb_of_neurons_layer_2

    def make_env(self, test):
        env = gym.make(self.env_name)
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - 0 if test else 0
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        return env

    def train(self):
        if self.put_seed==True:
            # Set a random seed used in PFRL
            utils.set_random_seed(self.seed)

        env = self.make_env(test=False)

        obs_space = env.observation_space
        action_space = env.action_space
        print("Observation space:", obs_space)
        print("Action space:", action_space)
        print("Dissipation : ", env.c)

        obs_size = obs_space.low.size
        action_size = action_space.low.size
        
        shape_policy = [obs_size, self.nb_of_neurons_layer_1, 
                        self.nb_of_neurons_layer_2, 1]
        shape_qfunc = [obs_size+action_size, self.nb_of_neurons_layer_1, 
                        self.nb_of_neurons_layer_2, 1]
        
        policy = nets.policy(shape_qfunc, self.threshold)
        q_func_1 = nets.qfunc(shape_qfunc)
        q_func_2 = nets.qfunc(shape_qfunc)
        
        policy_optimizer = torch.optim.Adam(policy.parameters())
        q_func_1_optimizer = torch.optim.Adam(q_func_1.parameters())
        q_func_2_optimizer = torch.optim.Adam(q_func_2.parameters())

        rbuf = replay_buffers.ReplayBuffer(self.rb_size)

        explorer = explorers.AdditiveGaussian(scale=0.1, low=env.action_space.low, high=env.action_space.high)

        def burnin_action_func():
            """Select random actions until model is updated one or more times."""
            return np.random.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

        def target_policy_smoothing_func(batch_action):
            """Add noises to actions for target policy smoothing."""
            noise = torch.clamp(0.2 * torch.randn_like(batch_action), -self.tar_act_noise, self.tar_act_noise)
            return torch.clamp(batch_action + noise, env.action_space.low[0], env.action_space.high[0])

        agent = pfrl.agents.TD3(
            policy,
            q_func_1,
            q_func_2,
            policy_optimizer,
            q_func_1_optimizer,
            q_func_2_optimizer,
            rbuf,
            gamma=self.gamma,
            soft_update_tau=self.tau,
            explorer=explorer,
            replay_start_size=self.replay_start_size,
            gpu=self.gpu,
            minibatch_size=self.minibatch_size,
            burnin_action_func=burnin_action_func,
            target_policy_smoothing_func=target_policy_smoothing_func,
        )

        if self.dirname_for_loading is not None and self.wrong_keys==True:
            policy_weights = torch.load(os.path.join(self.path_for_loading, './{}/best/policy.pt'.format(self.dirname_for_loading)))
            q_func_1_weights = torch.load(os.path.join(self.path_for_loading, './{}/best/q_func1.pt'.format(self.dirname_for_loading)))
            q_func_2_weights = torch.load(os.path.join(self.path_for_loading, './{}/best/q_func2.pt'.format(self.dirname_for_loading)))
            with torch.no_grad():
                policy.fc1_policy.weight.copy_(policy_weights['0.weight'])
                policy.fc1_policy.bias.copy_(policy_weights['0.bias'])
                policy.fc2_policy.weight.copy_(policy_weights['2.weight'])
                policy.fc2_policy.bias.copy_(policy_weights['2.bias'])
                policy.fc3_policy.weight.copy_(policy_weights['4.weight'])
                policy.fc3_policy.bias.copy_(policy_weights['4.bias'])

                q_func_1.fc1_qfunc.weight.copy_(q_func_1_weights['1.weight'])
                q_func_1.fc1_qfunc.bias.copy_(q_func_1_weights['1.bias'])
                q_func_1.fc2_qfunc.weight.copy_(q_func_1_weights['3.weight'])
                q_func_1.fc2_qfunc.bias.copy_(q_func_1_weights['3.bias'])
                q_func_1.fc3_qfunc.weight.copy_(q_func_1_weights['5.weight'])
                q_func_1.fc3_qfunc.bias.copy_(q_func_1_weights['5.bias'])

                q_func_2.fc1_qfunc.weight.copy_(q_func_2_weights['1.weight'])
                q_func_2.fc1_qfunc.bias.copy_(q_func_2_weights['1.bias'])
                q_func_2.fc2_qfunc.weight.copy_(q_func_2_weights['3.weight'])
                q_func_2.fc2_qfunc.bias.copy_(q_func_2_weights['3.bias'])
                q_func_2.fc3_qfunc.weight.copy_(q_func_2_weights['5.weight'])
                q_func_2.fc3_qfunc.bias.copy_(q_func_2_weights['5.bias'])

        elif self.dirname_for_loading is not None and self.wrong_keys==False:
            agent.load(os.path.join(self.path_for_loading, './{}/best'.format(self.dirname_for_loading)))

        eval_env = self.make_env(test=True)

        experiments.train_agent_with_evaluation(agent=agent,
                                                env=env,
                                                steps=self.steps,
                                                eval_env=eval_env,
                                                eval_n_steps=None,
                                                eval_n_episodes=self.eval_n_episodes,
                                                eval_interval=self.eval_interval,
                                                train_max_episode_len=self.train_max_episode_len,
                                                outdir=os.path.join(self.path_to_save, './{}'.format(self.dirname_to_save)),
                                                threshold=self.threshold,
                                                noise=self.noise,
                                                scheduler=self.scheduler,
                                                automatically_stop=self.automatically_stop,
                                                success_threshold=self.success_threshold,
                                                save_force_signal=self.save_force_signal,
                                                )
        return agent, eval_env

def run():
    

if __name__ == "__main__":
    run()

import numpy as np
import logging
import os
import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
import torch
from torch import nn
import gym
import gym_systmemoire
import gym_systmemoire.envs
import pandas as pd
import plotly.graph_objects as go
import Config_env
import networks as nets

# assert torch.cuda.is_available()
device = torch.device('cpu')

class Train():
    def __init__(self, exp_dir, seed, put_seed, gpu, env_name, gamma, rb_size, tau, replay_start_size, minibatch_size, steps, eval_n_episodes, 
                 eval_interval, train_max_episode_len, tar_act_noise, threshold=None, noise=None, scheduler=None, automatically_stop=False,
                 success_threshold=0.95, save_force_signal=False, wrong_keys=True, int_layers = [400, 300]):
        self.exp_dir = exp_dir
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
        self.tar_act_noise = tar_act_noise
        self.threshold = threshold
        self.noise = noise
        self.scheduler = scheduler
        self.automatically_stop = automatically_stop
        self.success_threshold = success_threshold
        self.save_force_signal = save_force_signal
        self.wrong_keys = wrong_keys
        self.int_layers = int_layers

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

    def main(self):
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
        
        shape_policy = [obs_size] + self.int_layers + [action_size]
        shape_qfunc = [obs_size+action_size] + self.int_layers + [1]
        
        policy = nets.policy(shape_qfunc, env.action_space.high[0])
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

        eval_env = self.make_env(test=True)

        experiments.train_agent_with_evaluation(agent=agent,
                                                env=env,
                                                steps=self.steps,
                                                eval_env=eval_env,
                                                eval_n_steps=None,
                                                eval_n_episodes=self.eval_n_episodes,
                                                eval_interval=self.eval_interval,
                                                train_max_episode_len=self.train_max_episode_len,
                                                outdir=self.exp_dir,
                                                )
        return agent, eval_env




class Test_phase():

    def __init__(self, gpu, env_name, env_surname, n_steps, n_episodes, max_episode_len, add_simu_time, t_simu, file_name, path_to_save, dirname_to_save,
                 path_for_loading, dirname_for_loading, c, init_state, target_state, save_obs, wrong_keys=True):
        self.gpu = gpu
        self.env_name = env_name
        self.env_surname = env_surname
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.max_episode_len = max_episode_len
        self.add_simu_time = add_simu_time
        self.t_simu = t_simu
        self.file_name = file_name
        self.path_to_save = path_to_save
        self.dirname_to_save = dirname_to_save
        self.path_for_loading = path_for_loading
        self.dirname_for_loading = dirname_for_loading
        self.c = c
        self.init_state = init_state
        self.target_state = target_state
        self.save_obs = save_obs
        self.wrong_keys = wrong_keys

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

    def bin_to_pos(self, binstate, eq_pos):
        if binstate == 0:
            become = eq_pos[0]
        else:
            become = eq_pos[2]
        return become

    def goalbin_to_goalpos(self, eq_positions, init_state, nb_ressort):
        goalpos = np.zeros_like(init_state, dtype='float')
        goalpos_cum = np.zeros_like(init_state, dtype='float')
        for i in range(nb_ressort):
            goalpos[i] = self.bin_to_pos(init_state[i], eq_positions[i])
        goalpos_cum[0] = goalpos[0]
        for i in range(nb_ressort - 1):
            goalpos_cum[i + 1] = goalpos_cum[i] + goalpos[i + 1]
        goalpos_cum = np.append(goalpos_cum, np.zeros(nb_ressort))
        return goalpos_cum.tolist()

    def run_episodes(self, env, agent, logger=None,
    ):
        """Run multiple episodes and return returns."""
        assert (self.n_steps is None) != (self.n_episodes is None)

        logger = logger or logging.getLogger(__name__)
        scores = []
        terminate = False
        timestep = 0
        pos = [[] for _ in range(env.nb_ressort)]
        vel = [[] for _ in range(env.nb_ressort)]
        actions = []

        reset = True

        while not terminate:
            if reset:
                obs = env.reset()
                for _ in range(env.nb_ressort):
                    pos[_].append(obs[_])
                for _ in range(env.nb_ressort):
                    vel[_].append(obs[_ + env.nb_ressort])
                done = False
                test_r = 0
                episode_len = 0
                info = {}
            a = agent.act(obs)
            obs, r, done, info = env.step(a)
            actions.append(a)
            for _ in range(env.nb_ressort):
                pos[_].append(obs[_])
            for _ in range(env.nb_ressort):
                vel[_].append(obs[_ + env.nb_ressort])
            test_r += r
            episode_len += 1
            timestep += 1
            reset = done or episode_len == self.max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)
            if reset:
                logger.info(
                    "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
                )
                # As mixing float and numpy float causes errors in statistics
                # functions, here every score is cast to float.
                scores.append(float(test_r))
                if self.add_simu_time==True:
                    for i in range(self.t_simu):
                        obs, r, done, info = env.step((0.,))
                        for _ in range(env.nb_ressort):
                            pos[_].append(obs[_])
                        for _ in range(env.nb_ressort):
                            vel[_].append(obs[_ + env.nb_ressort])

                episode_len = np.array([episode_len])
                os.makedirs(os.path.join(self.path_to_save, './{}'.format(self.dirname_to_save)), exist_ok=True)
                with open(os.path.join(self.path_to_save, './{}/nb_of_steps.npy'.format(self.dirname_to_save)), 'a') as f:
                    np.savetxt(f, episode_len)
                positions = np.array(pos)
                velocities = np.array(vel)
                force = np.array(actions)
                if self.save_obs==True:
                    with open(os.path.join(self.path_to_save, './{}/pos_{}_{}.npy'.format(self.dirname_to_save, self.file_name, len(scores))), 'w') as f:
                        np.savetxt(f, positions)
                    with open(os.path.join(self.path_to_save, './{}/vel_{}_{}.npy'.format(self.dirname_to_save, self.file_name, len(scores))), 'w') as f:
                        np.savetxt(f, velocities)
                    with open(os.path.join(self.path_to_save, './{}/force_{}_{}.npy'.format(self.dirname_to_save, self.file_name, len(scores))), 'w') as f:
                        np.savetxt(f, force)
                episode_len = 0
                pos = [[] for _ in range(env.nb_ressort)]
                vel = [[] for _ in range(env.nb_ressort)]
                actions = []
            if self.n_steps is None:
                terminate = len(scores) >= self.n_episodes
            else:
                terminate = timestep >= self.n_steps
        # If all steps were used for a single unfinished episode
        if len(scores) == 0:
            scores.append(float(test_r))
            logger.info(
                "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
            )
        return scores

    def main(self):
        nb_ressort = np.size(Config_env.exp[self.env_surname]['system'])
        eq_positions = np.array([Config_env.exp[self.env_surname]['system'][k].x_e for k in range(nb_ressort)])
        Config_env.exp[self.env_surname]['c'] = self.c
        Config_env.exp[self.env_surname]['goal_state'] = self.target_state
        init_pos = self.goalbin_to_goalpos(eq_positions, self.init_state, nb_ressort)
        Config_env.exp[self.env_surname]['ini_pos'] = init_pos

        env = self.make_env(test=False)

        obs_space = env.observation_space
        action_space = env.action_space
        print("Observation space:", obs_space)
        print("Action space:", action_space)

        obs_size = obs_space.low.size
        action_size = action_space.low.size

        class Mul(nn.Module):
            def __init__(self):
                super(Mul, self).__init__()
                self.threshold = torch.tensor(env.action_space.high[0]).to(device)

            def forward(self, x):
                x = x * self.threshold
                return x

        class Policy(nn.Module):
            def __init__(self):
                super(Policy, self).__init__()
                self.fc1_policy = nn.Linear(obs_size, 400)
                self.fc2_policy = nn.Linear(400, 300)
                self.fc3_policy = nn.Linear(300, action_size)
                self.policy_deter = pfrl.policies.DeterministicHead()

                self.act1_policy = nn.ReLU()
                self.act2_policy = nn.Tanh()

            def forward(self, x):
                x = self.act1_policy(self.fc1_policy(x))
                x = self.act1_policy(self.fc2_policy(x))
                x = self.act2_policy(self.fc3_policy(x))
                x = self.policy_deter(x)
                return x

        class Qfunc(nn.Module):
            def __init__(self):
                super(Qfunc, self).__init__()
                self.concat = pfrl.nn.ConcatObsAndAction()
                self.fc1_qfunc = nn.Linear(obs_size + action_size, 400)
                self.fc2_qfunc = nn.Linear(400, 300)
                self.fc3_qfunc = nn.Linear(300, 1)

                self.act1_qfunc = nn.ReLU()

            def forward(self, x):
                x = self.concat(x)
                x = self.act1_qfunc(self.fc1_qfunc(x))
                x = self.act1_qfunc(self.fc2_qfunc(x))
                x = self.fc3_qfunc(x)
                return x

        policy = Policy()
        q_func_1 = Qfunc()
        q_func_2 = Qfunc()

        policy_optimizer = torch.optim.Adam(policy.parameters())
        q_func_1_optimizer = torch.optim.Adam(q_func_1.parameters())
        q_func_2_optimizer = torch.optim.Adam(q_func_2.parameters())

        rbuf = replay_buffers.ReplayBuffer(10 ** 6)

        explorer = explorers.AdditiveGaussian(scale=0.1, low=env.action_space.low, high=env.action_space.high)

        agent = pfrl.agents.TD3(
            policy,
            q_func_1,
            q_func_2,
            policy_optimizer,
            q_func_1_optimizer,
            q_func_2_optimizer,
            rbuf,
            gamma=0.99,
            soft_update_tau=5e-3,
            explorer=explorer,
            replay_start_size=10000,
            gpu=self.gpu,
            minibatch_size=100,
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

        if self.path_for_loading is not None and self.wrong_keys==False:
            agent.load(os.path.join(self.path_for_loading, './{}/best'.format(self.dirname_for_loading)))

        eval_env = self.make_env(test=True)

        with agent.eval_mode():
            self.run_episodes(env=eval_env, agent=agent, logger=None)



class Force_analysis():
    def __init__(self, path_for_loading, dirname_for_loading, nb_of_tests, c_gridsearch, transition_name, path_for_fig, fig_name):
        self.path_for_loading = path_for_loading
        self.dirname_for_loading = dirname_for_loading
        self.nb_of_tests = nb_of_tests
        self.c_gridsearch = c_gridsearch
        self.transition_name = transition_name
        self.path_for_fig = path_for_fig
        self.fig_name = fig_name

    def get_force_array(self):
        force_signal = [[] for _ in self.c_gridsearch]
        for c_index in range(len(self.c_gridsearch)):
            force_signal[c_index].append(np.loadtxt(os.path.join(self.path_for_loading, './{}/force_c_{}_{}_transition_{}_1.npy').format(self.dirname_for_loading, int(self.c_gridsearch[c_index]), int(str(self.c_gridsearch[c_index]).split('.')[1]),
                                                                                                                      self.transition_name)))
        force_signal = np.array(force_signal)
        return force_signal

    def get_max_shape(self, force_signal):
        shape = 0
        for i in range(force_signal.shape[0]):
            if force_signal[i][0].shape[0] > shape:
                shape = force_signal[i][0].shape[0]
        return shape

    def steps(self, max_shape):
        steps = np.linspace(0, max_shape, max_shape)
        return steps

    def get_DataFrame(self, force_signal, steps):
        d = {'force_signal': force_signal, 'steps': steps, 'c': self.c_gridsearch}
        data = pd.DataFrame(data=d['force_signal'], columns=d['steps'], index=d['c'])
        return data

    #To use if the signal is not deterministic (noise was added)
    def mean_sig(self, c_index):
        sig = []
        shape = 1000
        for i in range(self.nb_of_tests):
            sig_path = os.path.join(self.path_for_loading, './{}/force_c_{}_{}_transition_{}_{}.npy').format(self.dirname_for_loading, int(self.c_gridsearch[c_index]), int(str(self.c_gridsearch[c_index]).split('.')[1]),
                                                                                          self.transition_name, i+1)
            if np.loadtxt(sig_path).shape[0] < shape:
                shape = np.loadtxt(sig_path).shape[0]
        for i in range(self.nb_of_tests):
            sig_path = os.path.join(self.path_for_loading, './{}/force_c_{}_{}_transition_{}_{}.npy').format(self.dirname_for_loading, int(self.c_gridsearch[c_index]), int(str(self.c_gridsearch[c_index]).split('.')[1]),
                                                                                          self.transition_name, i+1)
            sig.append(np.loadtxt(sig_path)[:shape])
        sig = np.array(sig)
        return np.mean(sig, axis=0)

    def get_mean_force_array(self):
        force_signal = [[] for _ in self.c_gridsearch]
        for c_index in range(len(self.c_gridsearch)):
            force_signal[c_index].append(self.mean_sig(c_index))
        force_signal = np.array(force_signal)
        return force_signal

    def uniform_shape(self, force_signal, max_shape):
        force_signal_uniform_shape = np.empty((force_signal.shape[0], max_shape))
        for i in range(force_signal.shape[0]):
            if force_signal[i][0].shape[0] < max_shape:
                force_signal_uniform_shape[i, :] = np.append(force_signal[i][0], np.zeros(max_shape - force_signal[i][0].shape[0]) + np.nan)
            else:
                force_signal_uniform_shape[i, :] = force_signal[i][0]
        return force_signal_uniform_shape

    def plot(self, data):
        fig = go.Figure(data=go.Heatmap(z=data, colorscale='RdBu', zmin=-1., zmax=1.))
        fig.update_yaxes(autorange=True)
        fig.update_yaxes(tickfont_size=32)
        fig.update_xaxes(tickfont_size=32)
        fig.update_layout(xaxis_title='Number of steps',
                          yaxis_title='Dissipation (Kg/s)',
                          title=dict(text='Force signal',
                                     font=dict(
                                         family='CMU Serif Bold',
                                         size=42,
                                         color='#000000'
                                     )
                                     ),
                          font=dict(
                              family='CMU Serif Bold',
                              size=32
                          ),
                          font_color="black",
                          margin=dict(r=140),
                          height=600,
                          width=600,
                          )
        fig.write_image(os.path.join(self.path_for_fig, './{}_transition_{}.pdf'.format(self.fig_name, self.transition_name)))

    def main(self):
        if self.nb_of_tests==1:
            force_signal = self.get_force_array()
        else:
            force_signal = self.get_mean_force_array()
        maximal_shape = self.get_max_shape(force_signal=force_signal)
        force_signal = self.uniform_shape(force_signal, maximal_shape)
        steps = self.steps(max_shape=maximal_shape)
        data = self.get_DataFrame(force_signal, steps)
        self.plot(data)


class Injected_energy_analysis():
    def __init__(self, path_for_loading, dirname_for_loading, nb_of_tests, c_gridsearch, nb_ressort, transition_name, path_for_fig, fig_name):
        self.path_for_loading = path_for_loading
        self.dirname_for_loading = dirname_for_loading
        self.nb_of_tests = nb_of_tests
        self.c_gridsearch = c_gridsearch
        self.nb_ressort = nb_ressort
        self.transition_name = transition_name
        self.path_for_fig = path_for_fig
        self.fig_name = fig_name

    def get_force_pos_array(self):
        force_signal = [[] for _ in self.c_gridsearch]
        positions = [[[] for _ in range(self.nb_ressort)] for _ in self.c_gridsearch]
        for c_index in range(len(self.c_gridsearch)):
            for idx in range(self.nb_of_tests):
                force_signal[c_index].append(np.loadtxt(os.path.join(self.path_for_loading, './{}/force_c_{}_{}_transition_{}_{}.npy').format(self.dirname_for_loading, int(self.c_gridsearch[c_index]), int(str(self.c_gridsearch[c_index]).split('.')[1]),
                                                                                                                           self.transition_name, idx+1)))
                for spring_idx in range(self.nb_ressort):
                    positions[c_index][spring_idx].append(np.loadtxt(os.path.join(self.path_for_loading, './{}/pos_c_{}_{}_transition_{}_{}.npy').format(self.dirname_for_loading, int(self.c_gridsearch[c_index]), int(str(self.c_gridsearch[c_index]).split('.')[1]),
                                                                                                                                             self.transition_name, idx+1))[spring_idx])
        force_signal = np.array(force_signal)
        positions = np.array(positions)
        return force_signal, positions

    def get_max_shape(self, signal):
        shape = 0
        for idx in range(signal.shape[0]):
            if signal[idx].shape[0] > shape:
                shape = signal[idx].shape[0]
        return shape

    def steps(self, max_shape):
        steps = np.linspace(0, max_shape, max_shape)
        return steps

    def get_energy(self, force_signal, positions):
        injected_energy = []
        shape = 1000
        for idx in range(self.nb_of_tests):
            if force_signal[idx].shape[0] < shape:
                shape = force_signal[idx].shape[0]
        for idx in range(self.nb_of_tests):
            if self.nb_ressort==1:
                pos_tmp = positions[:, idx][0]
                injected_energy.append(force_signal[idx][:shape] * pos_tmp[:shape])
            else:
                shape_tmp = positions[:, idx][0].shape[0]
                pos_tmp = np.concatenate((positions[:, idx][0], positions[:, idx][1])).reshape(-1, shape_tmp)
                for idx_2 in range(self.nb_ressort-2):
                    pos_tmp = np.concatenate((pos_tmp.flatten(), positions[:, idx][idx_2+2])).reshape(-1, shape_tmp)
                diff_pos = np.diff(pos_tmp, axis=1)
                elongation = diff_pos[-1]
                injected_energy.append(force_signal[idx][:shape] * elongation[:shape])
        return np.mean(injected_energy, axis=0)

    def get_mat_energy(self, force_signal, positions):
        injected_energy = []
        for c_index in range(len(self.c_gridsearch)):
            injected_energy.append(self.get_energy(force_signal[c_index], positions[c_index]))
        return np.array(injected_energy)

    def uniform_shape(self, inj_energy, max_shape):
        inj_energy_uniform_shape = np.empty((inj_energy.shape[0], max_shape))
        for i in range(inj_energy.shape[0]):
            if inj_energy[i].shape[0] < max_shape:
                inj_energy_uniform_shape[i, :] = np.append(np.array(inj_energy[i]),
                                                           np.zeros(max_shape - inj_energy[i].shape[0]) + np.nan)
            else:
                inj_energy_uniform_shape[i, :] = np.array(inj_energy[i])
        return inj_energy_uniform_shape

    def get_DataFrame(self, injected_energy, steps):
        d = {'injected_energy': injected_energy, 'steps': steps, 'c': self.c_gridsearch}
        data = pd.DataFrame(data=d['injected_energy'], columns=d['steps'], index=d['c'])
        return data

    def plot(self, data):
        fig = go.Figure(data=go.Heatmap(z=data, colorscale='RdBu', zmin=-0.01, zmax=0.01))
        fig.update_yaxes(autorange=True)
        fig.update_yaxes(tickfont_size=32)
        fig.update_xaxes(tickfont_size=32)
        fig.update_layout(xaxis_title='Number of steps',
                          yaxis_title='Dissipation (Kg/s)',
                          title=dict(text='Injected energy',
                                     font=dict(
                                         family='CMU Serif Bold',
                                         size=42,
                                         color='#000000'
                                     )
                                     ),
                          font=dict(
                              family='CMU Serif Bold',
                              size=32
                          ),
                          font_color="black",
                          margin=dict(r=140),
                          height=600,
                          width=600,
                          )
        fig.write_image(os.path.join(self.path_for_fig, './{}_transition_{}.pdf'.format(self.fig_name, self.transition_name)))

    def main(self):
        force_signal, positions = self.get_force_pos_array()
        injected_energy = self.get_mat_energy(force_signal, positions)
        maximal_shape = self.get_max_shape(signal=injected_energy)
        injected_energy = self.uniform_shape(injected_energy, maximal_shape)
        steps = self.steps(max_shape=maximal_shape)
        data = self.get_DataFrame(injected_energy, steps)
        self.plot(data)











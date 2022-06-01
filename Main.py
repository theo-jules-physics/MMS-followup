import networks as nets
import torch

shape = [10, 10, 1]
threshold = 1
policy = nets.policy(shape, threshold)
q_func_1 = nets.qfunc(shape)
q_func_2 = nets.qfunc(shape)

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
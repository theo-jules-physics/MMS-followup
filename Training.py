import pytorch_lightning as pl
import utils.agent as uage
import utils.networks as unet
import gym
import gym_systmemoire
import gym_systmemoire.envs

class Trainer:
    def __init__(self, env, state_dim, action_dim, actor_shape, critic_shape, 
                 actor_lr=1e-4, critic_lr=1e-3, buffer_size=int(1e6), batch_size=128, 
                 gamma=0.99, tau=1e-3, warmup_steps=int(1e3), device='cpu'):
        self.env = env
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor_shape = actor_shape
        self.critic_shape = critic_shape
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.device = device
        
        self.actor = unet.Policy(unet.simple_mlp(actor_shape), self.env.action_space.high[0]).to(device)
        self.critic = unet.Qfunc(unet.simple_mlp(critic_shape)).to(device)
        
        self.ddpg = uage.DDPG(self.actor, self.critic, self.action_dim, buffer_size, batch_size, gamma, tau, actor_lr,
                              critic_lr, warmup_steps, device)
    
    def train(self, num_episodes=1000, max_steps_per_episode=200):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps_per_episode):
                # Collect experience and add to replay buffer
                action = self.ddpg.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.ddpg.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Update networks if there are enough experiences in the replay buffer
                self.ddpg.update()

                if done:
                    break

            print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}")

        self.env.close()

if __name__ == '__main__':
    
    env_name = 'env-3M'
    env = gym.make(env_name)
    assert isinstance(env, gym.wrappers.TimeLimit)
    env = env.env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
            
    actor_shape = [state_dim, 64, 64, action_dim]
    critic_shape = [state_dim + action_dim, 64, 64, 1]
    
    trainer = Trainer(env, state_dim, action_dim, actor_shape, critic_shape)
    trainer.train()

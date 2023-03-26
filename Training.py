import sys
sys.path.append('C:/Users/Theo/Documents/Github/MMS-followup/utils')
import agent as uage
import networks as unet
import gym
import gym_systmemoire
import gym_systmemoire.envs
import wandb
import time
from datetime import datetime
from tqdm import tqdm
from collections import deque
import numpy as np

class Trainer:
    def __init__(self, env, state_dim, action_dim, actor_shape, critic_shape, 
                 actor_lr=1e-4, critic_lr=1e-3, buffer_size=int(1e6), batch_size=1024, 
                 gamma=0.99, tau=1e-3, warmup_steps=int(1e5), device='cuda',
                 project_name=None, experiment_name=None, wandb_log=True):
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
        self.wandb_log = wandb_log
        
        self.actor = unet.Policy(unet.simple_mlp(actor_shape), self.env.action_space.high[0]).to(device)
        self.critic = unet.Qfunc(unet.simple_mlp(critic_shape)).to(device)
        
        self.ddpg = uage.DDPG(self.actor, self.critic, self.action_dim, buffer_size, batch_size, gamma, tau, actor_lr,
                              critic_lr, warmup_steps, device)

        if project_name is not None and experiment_name is not None and wandb_log:
            wandb.init(project=project_name, name=experiment_name)
            wandb.watch([self.actor, self.critic])

    
    def train(self, num_episodes=1000, max_steps_per_episode=200):
        start_time = datetime.now()
        episode_times = []
        episode_success = []
        success_rate = deque(maxlen=100)
        pbar = tqdm(total=num_episodes)
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_start_time = datetime.now()
            
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
            

            episode_time = datetime.now() - episode_start_time
            episode_times.append(episode_time)
            success = int(total_reward >= 0)
            episode_success.append(success)
            success_rate.append(success)
            mean_success_rate = np.mean(success_rate)
            pbar.update()
            pbar.set_description(f"Ep. {episode+1}/{num_episodes} | "
                                  f"Time: {datetime.now()-start_time} | "
                                  f"Ep. Time: {episode_time} \n "
                                  f"Success Rate: {mean_success_rate:.2%} | "
                                  f"Est. Time Left: {(datetime.now()-start_time)/((episode+1)/num_episodes) - (datetime.now()-start_time)}")
            print(success)
            print('\n')
            if self.wandb_log:
                wandb.log({"episode_reward": total_reward, "Success Rate": mean_success_rate})
        pbar.close()
                


        self.env.close()

if __name__ == '__main__':
    
    env_name = 'env-3M'
    env = gym.make(env_name)
    assert isinstance(env, gym.wrappers.TimeLimit)
    env = env.env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
            
    actor_shape = [state_dim, 400, 300, action_dim]
    critic_shape = [state_dim + action_dim, 400, 300, 1]
    project_name = 'DDPG_bistable_springs'
    experiment_name = 'test'
    num_episodes = int(2e5)
    
    trainer = Trainer(env, state_dim, action_dim, actor_shape, critic_shape, 
                      project_name=project_name, experiment_name=experiment_name, wandb_log=True)
    trainer.train(num_episodes=num_episodes)

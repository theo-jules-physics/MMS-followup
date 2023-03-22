import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Tuple

Experience = namedtuple("Experience", ["state", "action", "reward", "done", "next_state"])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([experience.state for experience in batch])
        actions = torch.FloatTensor([experience.action for experience in batch])
        rewards = torch.FloatTensor([experience.reward for experience in batch])
        dones = torch.FloatTensor([experience.done for experience in batch])
        next_states = torch.FloatTensor([experience.next_state for experience in batch])
        return states, actions, rewards, dones, next_states

class DDPG:
    def __init__(self, actor, critic, action_dim, buffer_size=int(1e6), batch_size=128, gamma=0.99, tau=1e-3,
                 actor_lr=1e-4, critic_lr=1e-3, warmup_steps=int(1e5), device='cpu'):
        
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def act(self, state, noise=0.1):
        if len(self.replay_buffer) < self.warmup_steps:
            action = np.random.randn(self.action_dim)
        else:         
            state = torch.FloatTensor(state).to(self.device)
            action = self.actor(state).cpu().data.numpy()
            action += noise * np.random.randn(self.action_dim)
        return np.clip(action, -1.0, 1.0)
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(Experience(state, action, reward, done, next_state))

    def compute_critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q = self.target_critic(next_states, next_actions)
            target_q = rewards + self.gamma * (1.-dones) * next_q
        q_values = self.critic(states, actions)
        return F.mse_loss(q_values, target_q)

    def compute_actor_loss(self, states):
        return -self.critic(states, self.actor(states)).mean()

    def soft_update_target_networks(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, env, num_episodes, max_steps_per_episode):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            for step in range(max_steps_per_episode):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.replay_buffer) > self.batch_size:
                    self.update()

                if done:
                    break

            print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}")

    def update(self):
        if len(self.replay_buffer) < self.warmup_steps:
            return
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.batch_size)

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss = self.compute_critic_loss(states, actions, rewards, dones, next_states)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss(states)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update_target_networks()

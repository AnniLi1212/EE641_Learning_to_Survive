import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from .base_agent import BaseAgent
from ..utils import ReplayBuffer
import os

def __init__(self, state_shape, action_space, config):
    super().__init__(state_shape, action_space)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize networks
    self.policy_net = DQNNetwork(
        state_shape, 
        self.num_actions,
        hidden_sizes=config.get('hidden_sizes', [128, 64])
    ).to(self.device)
    self.target_net = DQNNetwork(
        state_shape, 
        self.num_actions,
        hidden_sizes=config.get('hidden_sizes', [128, 64])
    ).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    
    # hyperparameters from config or default values
    self.batch_size = config.get('batch_size', 32)
    self.gamma = config.get('gamma', 0.99)
    self.epsilon = config.get('epsilon_start', 1.0)
    self.epsilon_end = config.get('epsilon_end', 0.01)
    self.epsilon_decay = config.get('epsilon_decay', 0.995)
    self.target_update = config.get('target_update', 10)
    self.memory_size = config.get('memory_size', 10000)
    
    # initialize optimizer
    self.optimizer = optim.Adam(
        self.policy_net.parameters(), 
        lr=config.get('learning_rate', 0.001)
    )

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_sizes=[128, 64]):
        super().__init__()
        
        # flattened input size
        flat_size = np.prod(input_shape)
        
        # network layers
        layers = []
        prev_size = flat_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # output layer
        layers.append(nn.Linear(prev_size, num_actions))
        
        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

class DQNAgent(BaseAgent):
    def __init__(self, state_shape, action_space, config):
        """
        Args:
            state_shape: Shape of the observation space
            action_space: Gymnasium action space object
            config: Configuration dictionary containing hyperparameters
        """
        super().__init__(state_shape, action_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # initialize networks
        self.policy_net = DQNNetwork(state_shape, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(state_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # hyperparameters from config
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.target_update = config['target_update']
        self.memory_size = config['memory_size']
        
        # initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        
        # initialize replay memory
        self.memory = ReplayBuffer(self.memory_size)
        
        # initialize step counter
        self.steps_done = 0
        
    # select an action using epsilon-greedy policy
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # random action
            return self.action_space.sample()
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self, state, action, reward, next_state, done):
        # store transition in memory
        self.memory.push(state, action, reward, next_state, done)
        
        # update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if len(self.memory) < self.batch_size:
            return
        
        # sample batch using the ReplayBuffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device)
        
        # compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            dones = dones.bool()
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values
            
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path, _use_new_zipfile_serialization=True)
        
    def load(self, path):
        try:
            checkpoint = torch.load(path, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
        except RuntimeError as e:
            print("Warning: Using legacy loading method")
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
from collections import deque
import random
import numpy as np
import torch

# experience replay buffer for storing and sampling transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    # add a transition to the buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    # sample a batch of transitions
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        return states, actions, rewards, next_states, dones
    
    # get the number of transitions in the buffer
    def __len__(self):
        return len(self.buffer)

from collections import deque
import random
import numpy as np
import torch

# experience replay buffer for storing and sampling transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    # add a transition to the buffer
    def push(self, state, action, reward, next_state, done, additional_state, next_additional_state):
        self.buffer.append((state, action, reward, next_state, done, 
                          additional_state, next_additional_state))
    
    # sample a batch of transitions
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, add_states, next_add_states = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        add_states = torch.FloatTensor(add_states).to(device)
        next_add_states = torch.FloatTensor(next_add_states).to(device)
        
        return states, actions, rewards, next_states, dones, add_states, next_add_states
    
    def __len__(self):
        return len(self.buffer)

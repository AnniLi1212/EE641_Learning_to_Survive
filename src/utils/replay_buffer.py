from collections import deque
import random
import numpy as np
import torch

# experience replay buffer for storing and sampling transitions
class ReplayBuffer:
    def __init__(self, capacity, sequence_length=None):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        self.episode_buffer = []
    
    # add a transition to the buffer
    def push(self, state, action, reward, next_state, done, additional_state, next_additional_state):
        if self.sequence_length is None:
            self.buffer.append((state, action, reward, next_state, done, 
                                additional_state, next_additional_state))
        else:
            self.episode_buffer.append((state, action, reward, next_state, done, 
                                       additional_state, next_additional_state))
            if done:
                for i in range(len(self.episode_buffer) - self.sequence_length + 1):
                    sequence = self.episode_buffer[i:i+self.sequence_length]
                    self.buffer.append(sequence)
                self.episode_buffer = []
    
    # sample a batch of transitions
    def sample(self, batch_size, device):
        if self.sequence_length is None:
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones, add_states, next_add_states = zip(*batch)
        
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
            add_states = np.array(add_states)
            next_add_states = np.array(next_add_states)

        else:
            # rnn seq samples
            sequences = random.sample(self.buffer, batch_size)
            states_batch = []
            actions_batch = []
            rewards_batch = []
            next_states_batch = []
            dones_batch = []
            add_states_batch = []
            next_add_states_batch = []

            for sequence in sequences:
                # last transition in seq
                last_transition = sequence[-1]

                states_seq = [t[0] for t in sequence]
                add_states_seq = [t[5] for t in sequence]

                states_batch.append(states_seq)
                actions_batch.append(last_transition[1])
                rewards_batch.append(last_transition[2])
                next_states_batch.append(last_transition[3])
                dones_batch.append(last_transition[4])
                add_states_batch.append(add_states_seq)
                next_add_states_batch.append(last_transition[6])
            
            states = np.array(states_batch)
            actions = np.array(actions_batch)
            rewards = np.array(rewards_batch)
            next_states = np.array(next_states_batch)
            dones = np.array(dones_batch)
            add_states = np.array(add_states_batch)
            next_add_states = np.array(next_add_states_batch)

        return (
            torch.FloatTensor(states).to(device), 
            torch.LongTensor(actions).to(device), 
            torch.FloatTensor(rewards).to(device), 
            torch.FloatTensor(next_states).to(device), 
            torch.FloatTensor(dones).to(device), 
            torch.FloatTensor(add_states).to(device), 
            torch.FloatTensor(next_add_states).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

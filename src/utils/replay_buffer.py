from collections import deque
import random
import numpy as np
import torch

# experience replay buffer for storing and sampling transitions
class ReplayBuffer:
    def __init__(self, capacity, sequence_length=None, device='cpu'):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        self.device = device
        self.episode_buffer = []
    
    # add a transition to the buffer
    def push(self, state, action, reward, next_state, done, additional_state, next_additional_state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
        if isinstance(additional_state, (list, np.ndarray)):
            additional_state = torch.FloatTensor(additional_state)
        if isinstance(next_additional_state, (list, np.ndarray)):
            next_additional_state = torch.FloatTensor(next_additional_state)
        
        state = state.cpu()
        next_state = next_state.cpu()
        additional_state = additional_state.cpu()
        next_additional_state = next_additional_state.cpu()

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
        
            return (
                torch.stack(states).to(device),
                torch.LongTensor(actions).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.stack(next_states).to(device),
                torch.FloatTensor(dones).to(device),
                torch.stack(add_states).to(device),
                torch.stack(next_add_states).to(device)
            )

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

                states_seq = torch.stack([t[0] for t in sequence])
                add_states_seq = torch.stack([t[5] for t in sequence])

                states_batch.append(states_seq)
                actions_batch.append(last_transition[1])
                rewards_batch.append(last_transition[2])
                next_states_batch.append(last_transition[3])
                dones_batch.append(last_transition[4])
                add_states_batch.append(add_states_seq)
                next_add_states_batch.append(last_transition[6])
            
            return (
                torch.stack(states_batch).to(device),
                torch.LongTensor(actions_batch).to(device),
                torch.FloatTensor(rewards_batch).to(device),
                torch.stack(next_states_batch).to(device),
                torch.FloatTensor(dones_batch).to(device),
                torch.stack(add_states_batch).to(device),
                torch.stack(next_add_states_batch).to(device)
            )
    
    def __len__(self):
        return len(self.buffer)

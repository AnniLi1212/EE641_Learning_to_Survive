import torch
import torch.nn as nn
import torch.optim as optim
import random
from ..utils import ReplayBuffer
import os
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_sizes=[256, 128], rnn_hidden_size=None):
        super().__init__()
        # use conv to process input: 5 channels (one hot[wall, agent, cave], food value, threat attack)
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # calculate conv output size
        # input_shape[1] * input_shape[2]: size of the grid
        conv_out_size = input_shape[1] * input_shape[2] * 32
        additional_state_size = 3  # agent's health, hunger, attack
        self.flat_size = conv_out_size + additional_state_size

        self.rnn_hidden_size = rnn_hidden_size
        if self.rnn_hidden_size is not None:
            self.lstm = nn.LSTM(
                input_size=self.flat_size, 
                hidden_size=self.rnn_hidden_size, 
                num_layers=1, 
                batch_first=True
            )
            prev_size = self.rnn_hidden_size
        else:
            self.lstm = None
            prev_size = self.flat_size

        layers = []
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_actions)
    
    def init_hidden(self, batch_size, device):
        if self.rnn_hidden_size is not None:
            return (torch.zeros(1, batch_size, self.rnn_hidden_size, device=device),
                    torch.zeros(1, batch_size, self.rnn_hidden_size, device=device))
        return None

    def forward(self, x, additional_state, hidden_state=None):
        batch_size = x.size(0)
        if len(x.shape) == 5: # batch, seq, channel, grid_size, grid_size
            seq_len = x.size(1)
            x = x.view(batch_size * seq_len, *x.shape[2:])

            # cnn
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.flatten(1)

            if len(additional_state.shape) == 3: # batch, seq, features
                additional_state = additional_state.view(batch_size * seq_len, -1)

            # concat internal states
            x = torch.cat([x, additional_state], dim=1)
            x = x.view(batch_size, seq_len, -1)

            # rnn
            if self.lstm is not None:
                x, hidden_state = self.lstm(x, hidden_state)
                x = x[:, -1]
            else:
                x = x[:, -1]
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.flatten(1)
            x = torch.cat([x, additional_state], dim=1)

            if self.lstm is not None:
                x = x.unsqueeze(1)
                x, hidden_state = self.lstm(x, hidden_state)
                x = x.squeeze(1)

        x = self.network(x)
        return self.output(x), hidden_state

class DQNAgent:
    def __init__(self, state_shape, action_space, config):
        self.state_shape = state_shape
        self.action_space = action_space
        self.num_actions = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # hyperparameters from config
        self.batch_size = config.get('batch_size', 128)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.hidden_sizes = config.get('hidden_sizes', [256, 128])
        self.epsilon_decay = config.get('epsilon_decay', 0.997)
        self.target_update = config.get('target_update', 10)
        self.memory_size = config.get('memory_size', 100000)
        self.observation_range = config.get('observation_range', 10)

        self.num_grid_categories = 3 # wall, agent, cave
        self.food_value_max = config.get('food_value_max', 30)
        self.threat_attack_max = config.get('threat_attack_max', 40)
        self.agent_attack_max = config.get('agent_attack_max', 45)
        self.modified_state_shape = (self.num_grid_categories + 2,) + state_shape[1:]

        self.rnn_hidden_size = config.get('rnn_hidden_size', None)
        self.sequence_length = config.get('sequence_length', None)

        # initialize networks
        self.policy_net = DQNNetwork(
            self.modified_state_shape, 
            self.num_actions,
            self.hidden_sizes,
            self.rnn_hidden_size
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            self.modified_state_shape, 
            self.num_actions,
            self.hidden_sizes,
            self.rnn_hidden_size
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate
        )
        
        # initialize replay memory
        self.memory = ReplayBuffer(self.memory_size, self.sequence_length)

        # initialize hidden state
        self.hidden_state = None

        # initialize step counter
        self.steps_done = 0

    def _preprocess_single_state(self, state):
        grid = state[0]
        one_hot = torch.zeros((self.num_grid_categories, state.shape[1], state.shape[2]), 
                              device=self.device)
        
        for i in range(self.num_grid_categories):
            one_hot[i] = (grid == i+1).float() # skip 0=wall

        food_channel = state[1] / self.food_value_max
        threat_channel = state[2] / self.threat_attack_max

        processed_state = torch.cat([
            one_hot, 
            food_channel.unsqueeze(0), 
            threat_channel.unsqueeze(0)
        ], dim=0)

        return processed_state
    
    def _preprocess_state(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 5: # batchsize, seqlen, channel, grid_size, grid_size
            batch_size, seq_len = state.shape[0], state.shape[1]
            processed_batch = []
            for b in range(batch_size):
                processed_seq = []
                for t in range(seq_len):
                    s = state[b, t]
                    processed_seq.append(self._preprocess_single_state(s))
                processed_batch.append(torch.stack(processed_seq))

            return torch.stack(processed_batch)
        
        elif len(state.shape) == 4: # batchsize, channels, grid_size, grid_size
            batch_size = state.shape[0]
            processed_batch = []
            for b in range(batch_size):
                processed_batch.append(self._preprocess_single_state(state[b]))
            return torch.stack(processed_batch)
        
        elif len(state.shape) == 3: # channels, grid_size, grid_size
            return self._preprocess_single_state(state)
        
        else:
            raise ValueError(f"Error in state shape: {state.shape}")
    
    def get_q_values(self, state, info=None):
        if info is None:
            return None
        
        with torch.no_grad():
            processed_state = self._preprocess_state(state)
            state_tensor = processed_state.unsqueeze(0).to(self.device)
            additional_state = torch.FloatTensor([[
                info['health'] / 100.0, 
                info['hunger'] / 100.0, 
                info['attack'] / self.agent_attack_max
            ]]).to(self.device)
            q_values, _ = self.policy_net(state_tensor, additional_state)
            return q_values
    
    def select_action(self, state, info=None, training=True):
        # random action
        if training and random.random() < self.epsilon:
            action = self.action_space.sample()
            return action
        
        with torch.no_grad():
            # internal state normalized
            processed_state = self._preprocess_state(state)
            state_tensor = processed_state.unsqueeze(0).to(self.device)

            additional_state = torch.FloatTensor([
                [info['health'] / 100.0, 
                info['hunger'] / 100.0, 
                info['attack'] / self.agent_attack_max]
            ]).to(self.device)
            
            # calculate q values and select action
            q_values, self.hidden_state = self.policy_net(
                state_tensor, 
                additional_state, 
                self.hidden_state
            )
            action = q_values.argmax().item()
            return action
    
    def train(self, state, action, reward, next_state, done, info, next_info):
        # store transition in memory
        additional_state = [
            info['health'] / 100.0,
            info['hunger'] / 100.0,
            info['attack'] / self.agent_attack_max
        ]
        next_additional_state = [
            next_info['health'] / 100.0,
            next_info['hunger'] / 100.0,
            next_info['attack'] / self.agent_attack_max
        ]
        self.memory.push(state, action, reward, next_state, done, 
                        additional_state, next_additional_state)
    
        # update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if len(self.memory) < self.batch_size:
            return None
        
        # sample batch, calculate current q
        states, actions, rewards, next_states, dones, add_states, next_add_states = \
            self.memory.sample(self.batch_size, self.device)
        
        # process states
        if self.sequence_length is not None:
            if len(states.shape) == 4: # batch, channel, grid_size, grid_size
                states = states.unsqueeze(1)
                next_states = next_states.unsqueeze(1)

            if len(add_states.shape) == 2: # [batch, features]
                add_states = add_states.unsqueeze(1)
                next_add_states = next_add_states.unsqueeze(1)

        processed_states = self._preprocess_state(states)
        processed_next_states = self._preprocess_state(next_states)

        batch_hidden = self.policy_net.init_hidden(self.batch_size, self.device)
        current_q_values, _ = self.policy_net(processed_states, add_states, batch_hidden)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
    
        # calculate target q
        with torch.no_grad():
            next_q_values, _ = self.target_net(processed_next_states, next_add_states, batch_hidden)
            next_q_values = next_q_values.max(1)[0]
            next_q_values[dones.bool()] = 0.0
            target_q_values = rewards + self.gamma * next_q_values
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def reset(self):
        self.hidden_state = None
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': float(self.epsilon),
            'steps_done': int(self.steps_done),
            'state_shape': tuple(int(x) for x in self.state_shape),
            'num_actions': int(self.num_actions)
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = float(checkpoint['epsilon'])
        self.steps_done = int(checkpoint['steps_done'])